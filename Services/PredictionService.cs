using Microsoft.AspNetCore.Components.Forms;
using Microsoft.Extensions.Options;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using train_ml.Models;
using train_ml.Services;

namespace train_ml.Services
{
    public class PredictionService : IPredictionService
    {
        private readonly MLContext _mlContext;
        private ITransformer? _trainedModel;
        private const string TrainingDataPath = "Data/data1.csv";
        private readonly ThresholdOptions _thresholdOptions;
        private readonly Dictionary<string, float> _parameterThresholds;

        
        private static readonly Dictionary<string, string> _parameterDisplayNames = new(StringComparer.OrdinalIgnoreCase)
        {
            { "TS01", "Lưu lượng" },
            { "TS02", "Áp suất" },
            { "TS03", "Nhiệt độ" },
            { "TS04", "O₂ dư" },
            { "TS05", "Bụi tổng" },
            { "TS06", "CO" },
            { "TS07", "NOx" },
            { "TS08", "SO₂" }
        };

        public PredictionService(IOptions<ThresholdOptions> thresholdOptions)
        {
            _mlContext = new MLContext(seed: 0);

            _thresholdOptions = thresholdOptions?.Value ?? new ThresholdOptions();
            _parameterThresholds = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
            if (_thresholdOptions.ParameterCodes != null)
            {
                foreach (var kv in _thresholdOptions.ParameterCodes)
                    _parameterThresholds[kv.Key] = kv.Value;
            }

            // Huấn luyện mô hình khi khởi tạo Service
            TrainModel(TrainingDataPath);
        }

        private void TrainModel(string trainingDataPath)
        {
            // Kiểm tra xem file training có tồn tại không
            if (!File.Exists(trainingDataPath))
            {
                Console.WriteLine($"Lỗi: Không tìm thấy file dữ liệu huấn luyện tại {trainingDataPath}.");
                return;
            }

            //Tải dữ liệu
            IDataView trainingDataView = _mlContext.Data.LoadFromTextFile<PollutionData>(
                path: trainingDataPath,
                separatorChar: ',',
                hasHeader: true
            );

            //Định nghĩa Pipeline
            var pipeline = _mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Value")
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ParameterCodeEncoded", inputColumnName: "ParameterCode"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "UnitEncoded", inputColumnName: "Unit"))
                //Chuyển EntryDate thành feature
                .Append(_mlContext.Transforms.Text.FeaturizeText(outputColumnName: "EntryDateFeaturized", inputColumnName: "EntryDate"))
                //Kết hợp các cột feature
                .Append(_mlContext.Transforms.Concatenate("Features",
                    "EmissionSourceID",
                    "ParameterCodeEncoded",
                    "UnitEncoded",
                    "EntryDateFeaturized"
                    ))
                //Chọn thuật toán huấn luyện: FastTree Regression
                .Append(_mlContext.Regression.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));

            //Huấn luyện mô hình
            _trainedModel = pipeline.Fit(trainingDataView);

            //Đánh giá mô hình trên tập huấn luyện (Để tham khảo)
            var predictions = _trainedModel.Transform(trainingDataView);
            var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: "Label");

            Console.WriteLine($"[ML.NET] Mô hình đã được huấn luyện. R-Squared: {metrics.RSquared:F4}, MSE: {metrics.MeanSquaredError:F4}");
        }

        public PredictionResult UploadAndPredict(string filePath)
        {
            if (_trainedModel == null)
            {
                throw new InvalidOperationException("Mô hình chưa được huấn luyện. Vui lòng kiểm tra file training data.");
            }

            //Tải dữ liệu cần dự đoán
            IDataView dataView = _mlContext.Data.LoadFromTextFile<PollutionData>(
                path: filePath,
                separatorChar: ',',
                hasHeader: true
            );

            //Thực hiện dự đoán
            IDataView predictions = _trainedModel.Transform(dataView);

            //Đánh giá mô hình trên dữ liệu mới 
            var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: "Value");

            //Lấy kết quả dự đoán
            var predictionData = _mlContext.Data.CreateEnumerable<PollutionPrediction>(predictions, reuseRowObject: false).ToList();
            var originalData = _mlContext.Data.CreateEnumerable<PollutionData>(dataView, reuseRowObject: false).ToList();

            int warningCount = 0;

            var predictionRows = originalData.Zip(predictionData, (original, predicted) =>
            {
                
                float? threshold = null;
                if (!string.IsNullOrWhiteSpace(original.ParameterCode))
                {
                    if (_parameterThresholds.TryGetValue(original.ParameterCode, out var mapped))
                        threshold = mapped;
                }
                            
                bool isWarning = false;
                if (threshold.HasValue && threshold.Value > 0)
                {
                    isWarning = predicted.PredictedValue > threshold.Value;
                    if (isWarning) warningCount++;
                }

                
                string displayName = original.ParameterCode;
                if (!string.IsNullOrWhiteSpace(original.ParameterCode) &&
                    _parameterDisplayNames.TryGetValue(original.ParameterCode, out var mappedDisplay))
                {
                    displayName = mappedDisplay;
                }

                return new PredictionRow
                {
                    ParameterCode = original.ParameterCode,
                    ActualValue = original.Value,
                    ParameterDisplayName = displayName,
                    MeasurementDate = original.MeasurementDate,
                    PredictedValue = predicted.PredictedValue,
                    Unit = original.Unit,
                    IsWarning = isWarning,
                    Threshold = threshold
                };
            }).ToArray();

            // Trả về kết quả, bao gồm R2 và MSE
            return new PredictionResult
            {
                YearMonth = "File đã tải lên",
                MSE = metrics.MeanSquaredError,
                R2 = metrics.RSquared,
                WarningCount = warningCount,    
                Rows = predictionRows
            };
        }
    }
}