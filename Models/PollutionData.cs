using System;
using Microsoft.ML.Data;

namespace train_ml.Models
{
    public class PollutionData
    {
        [LoadColumn(0)] public float ResultID { get; set; }
        [LoadColumn(1)] public string ApprovedAt { get; set; }
        [LoadColumn(2)] public float EmissionSourceID { get; set; }
        [LoadColumn(3)] public string EntryDate { get; set; }
        [LoadColumn(4)] public float IsApproved { get; set; }
        [LoadColumn(5)] public DateTime MeasurementDate { get; set; }
        [LoadColumn(6)] public string ParameterCode { get; set; }
        [LoadColumn(7)] public string Remark { get; set; }
        [LoadColumn(8)] public string Unit { get; set; }
        [LoadColumn(9)] public float Value { get; set; } // label
        [LoadColumn(10)] public string Type { get; set; }

    }

    public class PollutionPrediction
    {
        [ColumnName("Score")]
        public float PredictedValue { get; set; }
    }

    public class PredictionRow
    {
        public string ParameterCode { get; set; }       
        public string ParameterDisplayName { get; set; }
        public DateTime MeasurementDate { get; set; }
        public float PredictedValue { get; set; }
        public string Unit { get; set; }
        public bool IsWarning { get; set; }
        public float? Threshold { get; set; }
        public float? ActualValue { get; set; }
    }

    public class PredictionResult
    {
        public string YearMonth { get; set; }
        public double MSE { get; set; }
        public double R2 { get; set; }
        public int WarningCount { get; set; }
        public PredictionRow[] Rows { get; set; }
    }
}