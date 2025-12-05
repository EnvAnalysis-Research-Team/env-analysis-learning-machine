using train_ml.Models;
using train_ml.Services;

var builder = WebApplication.CreateBuilder(args);

// Configure thresholds from configuration
builder.Services.Configure<ThresholdOptions>(builder.Configuration.GetSection("PollutionThresholds"));

// Thêm các dịch vụ vào container
builder.Services.AddControllersWithViews();
// Đăng ký Runtime Compilation (để chỉnh sửa View mà không cần build lại)
builder.Services.AddRazorPages().AddRazorRuntimeCompilation();

// Đăng ký Service với Singleton vì mô hình ML.NET chỉ cần train 1 lần
builder.Services.AddSingleton<IPredictionService, PredictionService>();

var app = builder.Build();

// Cấu hình HTTP request pipeline
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}
// Chạy Service để Huấn luyện mô hình ngay khi ứng dụng khởi động
app.Services.GetRequiredService<IPredictionService>();

app.UseStaticFiles();
app.UseRouting();
app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Pollution}/{action=Index}/{id?}"); // Đặt Pollution làm Controller mặc định

app.Run();