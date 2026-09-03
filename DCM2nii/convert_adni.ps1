# ============================================================
# ADNI DICOM -> NIfTI 批量转换脚本 (v3)
# 从 real_samples\ANDI\PatientXX_LABEL\ 转换到 data\LABEL\ 下
# 支持四分类: AD / NC / sMCI / pMCI
# ============================================================

# ---- 配置区：按需修改 ----
$dcm2niix   = "D:\AD_project\DCM2nii\dcm2niix.exe"   # dcm2niix.exe 的完整路径
$inputRoot  = "D:\AD_project\ad_cnn_project\data\real_samples\ANDI"
$outputRoot = "D:\AD_project\ad_cnn_project\data"
$validLabels = @("AD", "NC", "sMCI", "pMCI")      # 更新：四分类
$logFile    = "D:\AD_project\ad_cnn_project\convert_log.txt"

# ---- 前置检查 ----
if (-not (Test-Path $dcm2niix)) {
    Write-Host "错误：找不到 dcm2niix.exe，请检查路径：$dcm2niix" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $inputRoot)) {
    Write-Host "错误：找不到输入目录：$inputRoot" -ForegroundColor Red
    exit 1
}

foreach ($label in $validLabels) {
    $dir = Join-Path $outputRoot $label
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

function Log($msg, $color = "White") {
    Write-Host $msg -ForegroundColor $color
    Add-Content -Path $logFile -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
}

Log "==================== 开始转换 (v3, 四分类) ====================" "Cyan"
Log "输入目录: $inputRoot"
Log "输出根目录: $outputRoot"
Log "类别: $($validLabels -join ', ')"

$patientFolders = Get-ChildItem -Path $inputRoot -Directory
$successCount = 0
$skipCount = 0
$failCount = 0

foreach ($patientFolder in $patientFolders) {
    $folderName = $patientFolder.Name   # 例如 Patient01_NC / Patient05_pMCI

    $parts = $folderName -split "_"
    $label = $parts[-1]                 # 取最后一段：AD / NC / sMCI / pMCI
    $patientId = ($parts[0..($parts.Length - 2)] -join "_")

    # 大小写容错：有人可能写成 smci/PMCI 之类，统一按标准大小写匹配
    $matchedLabel = $validLabels | Where-Object { $_ -ieq $label }
    if (-not $matchedLabel) {
        Log "跳过 [$folderName]：无法识别类别标签 '$label'（应为 AD/NC/sMCI/pMCI 之一）" "Yellow"
        $skipCount++
        continue
    }
    $label = $matchedLabel   # 统一成标准写法（如 smci -> sMCI）

    $outDir = Join-Path $outputRoot $label

    # 已转换过则跳过
    $existing = Get-ChildItem -Path $outDir -Filter "$patientId*.nii*" -ErrorAction SilentlyContinue
    if ($existing) {
        Log "跳过 [$folderName]：$patientId 已存在于 $label 文件夹，视为已转换过" "DarkGray"
        $skipCount++
        continue
    }

    # 递归找到所有实际包含 .dcm 文件的最底层目录
    $dicomDirs = Get-ChildItem -Path $patientFolder.FullName -Directory -Recurse |
                 Where-Object { (Get-ChildItem $_.FullName -Filter "*.dcm" -File -ErrorAction SilentlyContinue).Count -gt 0 }

    $rootDcm = Get-ChildItem -Path $patientFolder.FullName -Filter "*.dcm" -File -ErrorAction SilentlyContinue
    if ($rootDcm.Count -gt 0) {
        $dicomDirs = @($patientFolder) + $dicomDirs
    }

    if ($dicomDirs.Count -eq 0) {
        Log "失败 [$folderName]：在该文件夹下未找到任何 .dcm 文件" "Red"
        $failCount++
        continue
    }

    Log "处理中 [$folderName] -> 类别=$label, 找到 $($dicomDirs.Count) 个DICOM序列文件夹"

    $seriesIndex = 0
    foreach ($dcmDir in $dicomDirs) {
        $seriesIndex++
        if ($dicomDirs.Count -eq 1) {
            $outFileName = $patientId
        } else {
            $outFileName = "${patientId}_series${seriesIndex}"
        }

        $args = @(
            "-o", "`"$outDir`"",
            "-f", "`"$outFileName`"",   # 强制用患者ID命名，忽略DICOM协议名
            "-z", "y",
            "-b", "n",
            "-w", "2",
            "`"$($dcmDir.FullName)`""
        )

        $process = Start-Process -FilePath $dcm2niix -ArgumentList $args -NoNewWindow -Wait -PassThru `
                   -RedirectStandardOutput "$env:TEMP\dcm2niix_out.txt" `
                   -RedirectStandardError "$env:TEMP\dcm2niix_err.txt"

        if ($process.ExitCode -eq 0) {
            $producedFile = Get-ChildItem -Path $outDir -Filter "$outFileName*.nii*" -ErrorAction SilentlyContinue
            if ($producedFile) {
                Log "  成功: $($producedFile.Name)" "Green"
                $successCount++
            } else {
                Log "  警告：exitcode=0但未找到输出文件 ($outFileName)，可能该序列不是有效3D体积" "Yellow"
                $failCount++
            }
        } else {
            $errContent = Get-Content "$env:TEMP\dcm2niix_err.txt" -Raw -ErrorAction SilentlyContinue
            Log "  失败：转换出错 (序列: $($dcmDir.FullName))" "Red"
            if ($errContent) { Log "    错误详情: $errContent" "Red" }
            $failCount++
        }
    }
}

Log "==================== 转换完成 ====================" "Cyan"
Log "成功: $successCount   跳过: $skipCount   失败: $failCount" "Cyan"

# 按类别统计最终文件数量
Log "---- 各类别当前文件数量 ----"
foreach ($label in $validLabels) {
    $dir = Join-Path $outputRoot $label
    $count = (Get-ChildItem -Path $dir -Filter "*.nii*" -File -ErrorAction SilentlyContinue).Count
    Log "  $label : $count 个文件"
}

Log "详细日志已保存至: $logFile"