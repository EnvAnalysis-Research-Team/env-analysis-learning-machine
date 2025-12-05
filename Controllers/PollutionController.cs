using Microsoft.AspNetCore.Mvc;
using train_ml.Services;

namespace train_ml.Controllers
{
    public class PollutionController : Controller
    {
        private readonly IPredictionService _service;

        public PollutionController(IPredictionService service) => _service = service;

        [HttpGet]
        public IActionResult Index() => View();

        [HttpPost]
        public IActionResult UploadCsv(IFormFile csvFile)
        {
            if (csvFile == null || csvFile.Length == 0)
            {
                ModelState.AddModelError("csvFile", "Vui lòng chọn một file CSV.");
                return View("Index");
            }

            // Lưu file tạm thời để ML.NET có thể đọc
            var tempPath = Path.GetTempFileName();
            try
            {
                // Sao chép nội dung file upload vào file tạm thời
                using (var stream = new FileStream(tempPath, FileMode.Create))
                {
                    csvFile.CopyTo(stream);
                }

                // Thực hiện dự đoán và đánh giá
                var result = _service.UploadAndPredict(tempPath);

                // Chuyển sang View kết quả
                return View("Result", result);
            }
            catch (Exception ex)
            {
                // Xử lý lỗi
                ModelState.AddModelError("", $"Lỗi xử lý file hoặc dự đoán: {ex.Message}");
                return View("Index");
            }
            finally
            {
                // Xóa file tạm thời sau khi xử lý xong
                System.IO.File.Delete(tempPath);
            }
        }
    }
}