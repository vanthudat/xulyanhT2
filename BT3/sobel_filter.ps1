Add-Type -AssemblyName System.Drawing

# Ma tran I doc tu anh bai tap (10 x 10).
$I = @(
    @(0, 0, 0, 0, 200, 200, 0, 0, 0, 0),
    @(0, 0, 0, 0, 200, 200, 0, 0, 0, 0),
    @(0, 0, 0, 200, 200, 200, 200, 0, 0, 0),
    @(0, 0, 200, 200, 200, 200, 200, 200, 0, 0),
    @(200, 200, 200, 200, 200, 200, 200, 200, 200, 200),
    @(200, 200, 200, 200, 200, 200, 200, 200, 200, 200),
    @(0, 0, 200, 200, 200, 200, 200, 200, 0, 0),
    @(0, 0, 0, 200, 200, 200, 200, 0, 0, 0),
    @(0, 0, 0, 0, 200, 200, 0, 0, 0, 0),
    @(0, 0, 0, 0, 200, 200, 0, 0, 0, 0)
)

$Kx = @(
    @(-1, 0, 1),
    @(-2, 0, 2),
    @(-1, 0, 1)
)

$Ky = @(
    @(-1, -2, -1),
    @( 0,  0,  0),
    @( 1,  2,  1)
)

function New-ZeroMatrix([int]$Rows, [int]$Cols) {
    $matrix = @()
    for ($r = 0; $r -lt $Rows; $r++) {
        $matrix += ,([int[]](0..($Cols - 1) | ForEach-Object { 0 }))
    }
    return ,$matrix
}

function Invoke-Convolution3x3($InputMatrix, $Kernel) {
    $rows = $InputMatrix.Count
    $cols = $InputMatrix[0].Count
    $output = New-ZeroMatrix $rows $cols

    for ($y = 0; $y -lt $rows; $y++) {
        for ($x = 0; $x -lt $cols; $x++) {
            $sum = 0
            for ($ky = 0; $ky -lt 3; $ky++) {
                for ($kx = 0; $kx -lt 3; $kx++) {
                    $iy = $y + $ky - 1
                    $ix = $x + $kx - 1
                    if ($iy -ge 0 -and $iy -lt $rows -and $ix -ge 0 -and $ix -lt $cols) {
                        $sum += $InputMatrix[$iy][$ix] * $Kernel[$ky][$kx]
                    }
                }
            }
            $output[$y][$x] = $sum
        }
    }
    return ,$output
}

$Gx = Invoke-Convolution3x3 $I $Kx
$Gy = Invoke-Convolution3x3 $I $Ky
$Magnitude = New-ZeroMatrix 10 10
$maxMagnitude = 0.0
for ($y = 0; $y -lt 10; $y++) {
    for ($x = 0; $x -lt 10; $x++) {
        $value = [Math]::Sqrt(($Gx[$y][$x] * $Gx[$y][$x]) + ($Gy[$y][$x] * $Gy[$y][$x]))
        $Magnitude[$y][$x] = [int][Math]::Round($value)
        if ($value -gt $maxMagnitude) { $maxMagnitude = $value }
    }
}

function Matrix-ToText($Matrix) {
    $lines = @()
    foreach ($row in $Matrix) {
        $lines += (($row | ForEach-Object { '{0,4}' -f $_ }) -join ' ')
    }
    return ($lines -join [Environment]::NewLine)
}

$report = @"
SOBEL CHO MA TRAN ANH 1
Quy uoc: zero-padding, tich chap truc tiep voi kernel nhu ben duoi.

Kx = [-1  0  1; -2  0  2; -1  0  1]
Ky = [-1 -2 -1;  0  0  0;  1  2  1]

Gx:
$(Matrix-ToText $Gx)

Gy:
$(Matrix-ToText $Gy)

|G| = round(sqrt(Gx^2 + Gy^2)):
$(Matrix-ToText $Magnitude)
"@
Set-Content -LiteralPath (Join-Path $PSScriptRoot 'sobel_values.txt') -Value $report -Encoding UTF8

function Draw-MatrixPanel {
    param(
        [System.Drawing.Graphics]$Graphics,
        [string]$Title,
        $Matrix,
        [int]$OriginX,
        [int]$OriginY,
        [bool]$Heatmap,
        [double]$ScaleMax
    )

    $cellW = 61
    $cellH = 55
    $titleFont = New-Object System.Drawing.Font('Arial', 24, [System.Drawing.FontStyle]::Bold)
    $valueFont = New-Object System.Drawing.Font('Consolas', 14, [System.Drawing.FontStyle]::Regular)
    $center = New-Object System.Drawing.StringFormat
    $center.Alignment = [System.Drawing.StringAlignment]::Center
    $center.LineAlignment = [System.Drawing.StringAlignment]::Center
    $Graphics.DrawString($Title, $titleFont, [System.Drawing.Brushes]::Navy, $OriginX, $OriginY - 48)

    for ($r = 0; $r -lt 10; $r++) {
        for ($c = 0; $c -lt 10; $c++) {
            $x = $OriginX + $c * $cellW
            $y = $OriginY + $r * $cellH
            $rect = New-Object System.Drawing.Rectangle($x, $y, $cellW, $cellH)
            $v = [double]$Matrix[$r][$c]

            if ($Heatmap) {
                $level = if ($ScaleMax -gt 0) { [Math]::Min(255, [int][Math]::Round(255 * [Math]::Abs($v) / $ScaleMax)) } else { 0 }
                $fill = [System.Drawing.Color]::FromArgb(255, 255 - $level, 255 - [int]($level * 0.55), 255)
            } else {
                $gray = [Math]::Max(0, [Math]::Min(255, [int][Math]::Round($v * 255 / 200)))
                $fill = [System.Drawing.Color]::FromArgb(255, $gray, $gray, $gray)
            }
            $brush = New-Object System.Drawing.SolidBrush($fill)
            $Graphics.FillRectangle($brush, $rect)
            $Graphics.DrawRectangle([System.Drawing.Pens]::DimGray, $rect)
            $textColor = if ((-not $Heatmap -and $v -le 100) -or ($Heatmap -and $level -gt 170)) { [System.Drawing.Brushes]::White } else { [System.Drawing.Brushes]::Black }
            $textRect = New-Object System.Drawing.RectangleF([float]$x, [float]$y, [float]$cellW, [float]$cellH)
            $Graphics.DrawString(([int]$v).ToString(), $valueFont, $textColor, $textRect, $center)
            $brush.Dispose()
        }
    }
    $titleFont.Dispose()
    $valueFont.Dispose()
    $center.Dispose()
}

$bitmap = New-Object System.Drawing.Bitmap(1500, 1450)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$graphics.Clear([System.Drawing.Color]::White)

$mainFont = New-Object System.Drawing.Font('Arial', 34, [System.Drawing.FontStyle]::Bold)
$subFont = New-Object System.Drawing.Font('Arial', 18, [System.Drawing.FontStyle]::Regular)
$graphics.DrawString('KET QUA LOC SOBEL - MA TRAN ANH 1', $mainFont, [System.Drawing.Brushes]::DarkBlue, 355, 24)
$graphics.DrawString('Kx = [-1 0 1; -2 0 2; -1 0 1]    Ky = [-1 -2 -1; 0 0 0; 1 2 1]    Bien: zero-padding', $subFont, [System.Drawing.Brushes]::Black, 220, 78)

Draw-MatrixPanel $graphics 'Ma tran goc I' $I 85 170 $false 200
Draw-MatrixPanel $graphics 'Dao ham ngang Gx' $Gx 805 170 $true 800
Draw-MatrixPanel $graphics 'Dao ham doc Gy' $Gy 85 815 $true 800
Draw-MatrixPanel $graphics 'Do lon |G| (lam tron)' $Magnitude 805 815 $true $maxMagnitude

$footerFont = New-Object System.Drawing.Font('Arial', 16, [System.Drawing.FontStyle]::Italic)
$graphics.DrawString('|G| = round(sqrt(Gx^2 + Gy^2)); gia tri lon nhat = ' + ([int][Math]::Round($maxMagnitude)), $footerFont, [System.Drawing.Brushes]::DarkSlateGray, 470, 1395)

$outputPath = Join-Path $PSScriptRoot 'sobel_result.png'
$bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
$footerFont.Dispose()
$mainFont.Dispose()
$subFont.Dispose()
$graphics.Dispose()
$bitmap.Dispose()

# Anh bien Sobel doc lap: chuan hoa |G| ve [0, 255] va phong to theo nearest-neighbor.
$smallEdge = New-Object System.Drawing.Bitmap(10, 10)
for ($y = 0; $y -lt 10; $y++) {
    for ($x = 0; $x -lt 10; $x++) {
        $gray = [int][Math]::Round(255 * $Magnitude[$y][$x] / $maxMagnitude)
        $smallEdge.SetPixel($x, $y, [System.Drawing.Color]::FromArgb($gray, $gray, $gray))
    }
}
$edgeBitmap = New-Object System.Drawing.Bitmap(800, 800)
$edgeGraphics = [System.Drawing.Graphics]::FromImage($edgeBitmap)
$edgeGraphics.Clear([System.Drawing.Color]::Black)
$edgeGraphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::NearestNeighbor
$edgeGraphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::Half
$edgeGraphics.DrawImage($smallEdge, 0, 0, 800, 800)
$edgePath = Join-Path $PSScriptRoot 'sobel_edge_map.png'
$edgeBitmap.Save($edgePath, [System.Drawing.Imaging.ImageFormat]::Png)
$edgeGraphics.Dispose()
$edgeBitmap.Dispose()
$smallEdge.Dispose()

Write-Output "Created: $outputPath"
Write-Output "Created: $edgePath"
Write-Output "Maximum magnitude: $([int][Math]::Round($maxMagnitude))"
