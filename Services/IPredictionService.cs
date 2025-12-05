using train_ml.Models;

namespace train_ml.Services
{
    public interface IPredictionService
    {
        PredictionResult UploadAndPredict(string filePath);
    }
}