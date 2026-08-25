import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman
from evaluation.hrv import *
import os
import json
import time
import numpy as np
from tqdm import tqdm

def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr

def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    predict_sdnn_fft_all = list()
    gt_sdnn_fft_all = list()
    predict_rmssd_fft_all = list()
    gt_rmssd_fft_all = list()
    
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")

            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                SNR_all.append(SNR)

            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, sdnn_label, sdnn_pred ,rmssd_label, rmssd_pred= calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT',filename_id=index)
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
                predict_sdnn_fft_all.append(sdnn_pred)
                gt_sdnn_fft_all.append(sdnn_label)
                predict_rmssd_fft_all.append(rmssd_pred)
                gt_rmssd_fft_all.append(rmssd_label)
            else:
                raise ValueError("Inference evaluation method name wrong!")

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_hr_fft_all)
        predict_sdnn_valid = np.array(predict_sdnn_fft_all)
        gt_sdnn_valid = np.array(gt_sdnn_fft_all)
        predict_rmssd_valid = np.array(predict_rmssd_fft_all)
        gt_rmssd_valid = np.array(gt_rmssd_fft_all)

        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            elif metric == "RMSE":
                RMSE_FFT = np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all)))
                standard_error = np.std(np.square(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif metric == "SDNN":
                SDNN_MAE = np.mean(np.abs(predict_sdnn_valid - gt_sdnn_valid))
                standard_error = np.std(np.abs(predict_sdnn_valid - gt_sdnn_valid)) / np.sqrt(num_test_samples)

                predict_sdnn_mean = np.mean(predict_sdnn_valid)
                gt_sdnn_mean = np.mean(gt_sdnn_valid)

                print("SDNN MAE: {0} +/- {1}".format(SDNN_MAE, standard_error))
                print("  Predict SDNN mean: {0}, Ground Truth SDNN mean: {1}".format(predict_sdnn_mean, gt_sdnn_mean))

            elif metric == "RMSSD":
                RMSSD_MAE = np.mean(np.abs(predict_rmssd_valid - gt_rmssd_valid))
                standard_error = np.std(np.abs(predict_rmssd_valid - gt_rmssd_valid)) / np.sqrt(num_test_samples)

                predict_rmssd_mean = np.mean(predict_rmssd_valid)
                gt_rmssd_mean = np.mean(gt_rmssd_valid)

                print("RMSSD MAE: {0} +/- {1}".format(RMSSD_MAE, standard_error))
                print("  Predict RMSSD mean: {0}, Ground Truth RMSSD mean: {1}".format(predict_rmssd_mean, gt_rmssd_mean))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")

        # ============================
        # [ADDED] Save metrics (non-intrusive)
        # ============================
        save_root = getattr(config.TEST, "METRICS_SAVE_DIR", "./saved_metrics")
        save_dir = os.path.join(save_root, filename_id)
        os.makedirs(save_dir, exist_ok=True)  # exist_ok avoids error if exists [web:23]

        stamp = time.strftime("%Y%m%d-%H%M%S")
        npz_path = os.path.join(save_dir, f"{filename_id}_arrays_{stamp}.npz")
        json_path = os.path.join(save_dir, f"{filename_id}_summary_{stamp}.json")

        # 1) Save all raw arrays in one .npz (multiple arrays in one file) [web:17]
        np.savez(
            npz_path,
            gt_hr_fft_all=gt_hr_fft_all,
            predict_hr_fft_all=predict_hr_fft_all,
            snr_all=SNR_all,
            gt_sdnn_valid=gt_sdnn_valid,
            predict_sdnn_valid=predict_sdnn_valid,
            gt_hr_peak_all=np.array(gt_hr_peak_all),
            predict_hr_peak_all=np.array(predict_hr_peak_all),
        )

        # 2) Save a lightweight summary json (so you can quickly grep results) [web:22]
        summary = {
            "filename_id": filename_id,
            "toolbox_mode": str(config.TOOLBOX_MODE),
            "evaluation_method": str(config.INFERENCE.EVALUATION_METHOD),
            "dataset": str(getattr(config.TEST.DATA, "DATASET", "")),
            "fs": float(config.TEST.DATA.FS),
            "num_samples_fft": int(len(predict_hr_fft_all)),
            "num_samples_peak": int(len(predict_hr_peak_all)),
            "saved_npz": npz_path,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)  # keep utf-8 chars [web:22]

        print(f"[Saved] metrics arrays -> {npz_path}")
        print(f"[Saved] metrics summary -> {json_path}")



    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        SNR_all = np.array(SNR_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                RMSE_PEAK = np.sqrt(np.mean(np.square(predict_hr_peak_all - gt_hr_peak_all)))
                standard_error = np.std(np.square(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")
    


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data

