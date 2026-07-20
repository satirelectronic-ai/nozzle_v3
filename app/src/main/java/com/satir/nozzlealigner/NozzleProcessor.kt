package com.satir.nozzlealigner

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.core.Core
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot

/**
 * Görüntü işleme:
 *  - Ekran SİYAH-BEYAZ gösterilir.
 *  - Sadece KIRMIZI lazer ışığı (renk/HSV bazlı) algılanır ve kırmızı vurgulanır.
 *  - Kırmızı noktanın SABİT merkeze göre sapması ölçülür (yön + mm).
 */
class NozzleProcessor {

    data class Result(
        val ok: Boolean,
        val frameW: Int,
        val frameH: Int,
        val nozzleCenter: Point?,
        val nozzleRadiusPx: Double,
        val beamCenter: Point?,
        val refX: Double,
        val refY: Double,
        val offsetPx: Double,
        val offsetMm: Double,
        val dx: Double,
        val dy: Double,
        val direction: String,
        val state: State,
        val display: Bitmap?     // siyah-beyaz + kırmızı vurgu görüntüsü
    )

    enum class State { GREEN, YELLOW, RED, NONE }

    // Ayarlanabilir parametreler
    @Volatile var beamThreshold = 200     // kırmızı için min PARLAKLIK (V) — ten elenir
    @Volatile var minSat = 150            // min DOYGUNLUK (S) — mat ten elenir
    @Volatile var maxBlobArea = 4000.0    // bundan büyük leke = ten/arka plan, elenir
    @Volatile var minNozzleRadius = 40
    @Volatile var maxNozzleRadius = 400
    @Volatile var mmPerPx = 0.0
    @Volatile var greenThreshMm = 0.03
    @Volatile var yellowThreshMm = 0.10
    @Volatile var greenThreshPx = 3.0
    @Volatile var yellowThreshPx = 10.0

    // Sabit merkez referansı (0..1)
    @Volatile var refFracX = 0.5f
    @Volatile var refFracY = 0.5f

    // Hedef merkez modu: AUTO=daire, MANUAL=elle nişan, AUTO_THEN_MANUAL=daire yoksa nişan
    enum class CenterMode { AUTO, MANUAL, AUTO_THEN_MANUAL }
    @Volatile var centerMode = CenterMode.AUTO_THEN_MANUAL

    var invertX = false
    var invertY = false

    // Nozul tespiti (HoughCircles) pahalı; her karede değil aralıklı çalıştır.
    private var frameCount = 0
    private var lastNozzle: Pair<Point, Double>? = null

    // Yeniden kullanılan Mat'ler
    private var yuv: Mat? = null
    private var rgba: Mat? = null
    private var rgbaRot: Mat? = null
    private var gray: Mat? = null
    private var rgb: Mat? = null
    private var hsv: Mat? = null
    private var mask: Mat? = null
    private var mask2: Mat? = null
    private var display: Mat? = null
    private var work: Mat? = null

    fun process(data: ByteArray, width: Int, height: Int, rotationDegrees: Int = 0): Result {
        val yuvMat = ensure(yuv, height + height / 2, width, CvType.CV_8UC1).also { yuv = it }
        yuvMat.put(0, 0, data)

        // Renkli görüntü (kırmızı tespiti + gösterim için)
        var color = rgba ?: Mat().also { rgba = it }
        Imgproc.cvtColor(yuvMat, color, Imgproc.COLOR_YUV2RGBA_NV21)

        var w = width; var h = height
        if (rotationDegrees != 0) {
            val cr = rgbaRot ?: Mat().also { rgbaRot = it }
            when (rotationDegrees) {
                90 -> { Core.rotate(color, cr, Core.ROTATE_90_CLOCKWISE); w = height; h = width; color = cr }
                180 -> { Core.rotate(color, cr, Core.ROTATE_180); color = cr }
                270 -> { Core.rotate(color, cr, Core.ROTATE_90_COUNTERCLOCKWISE); w = height; h = width; color = cr }
            }
        }

        // Gri (nozul çemberi + gösterim tabanı)
        val g = gray ?: Mat().also { gray = it }
        Imgproc.cvtColor(color, g, Imgproc.COLOR_RGBA2GRAY)

        // --- LAZER maskesi: parlak kırmızı HALE + aşırı parlak beyaz ÇEKİRDEK ---
        val rgbM = rgb ?: Mat().also { rgb = it }
        Imgproc.cvtColor(color, rgbM, Imgproc.COLOR_RGBA2RGB)
        val hsvM = hsv ?: Mat().also { hsv = it }
        Imgproc.cvtColor(rgbM, hsvM, Imgproc.COLOR_RGB2HSV)

        val m1 = mask ?: Mat().also { mask = it }
        val m2 = mask2 ?: Mat().also { mask2 = it }
        val vMin = beamThreshold.toDouble()       // min parlaklık (V)
        val sMin = minSat.toDouble()              // min doygunluk (S) — mat teni eler
        // 1) Parlak, DOYGUN kırmızı hale (H iki uçta), yüksek S ve V
        Core.inRange(hsvM, Scalar(0.0, sMin, vMin), Scalar(12.0, 255.0, 255.0), m1)
        Core.inRange(hsvM, Scalar(168.0, sMin, vMin), Scalar(179.0, 255.0, 255.0), m2)
        Core.add(m1, m2, m1)
        // 2) Lazerin doymuş BEYAZ çekirdeği: çok parlak + düşük doygunluk
        Core.inRange(hsvM, Scalar(0.0, 0.0, 250.0), Scalar(179.0, 60.0, 255.0), m2)
        Core.add(m1, m2, m1)
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.morphologyEx(m1, m1, Imgproc.MORPH_OPEN, k)
        Imgproc.morphologyEx(m1, m1, Imgproc.MORPH_CLOSE, k)   // çekirdek+haleyi birleştir

        val beam = bestLaserBlob(m1)

        // --- Gösterim: siyah-beyaz + kırmızı vurgu ---
        val disp = display ?: Mat().also { display = it }
        Imgproc.cvtColor(g, disp, Imgproc.COLOR_GRAY2RGBA)     // gri -> RGBA (hâlâ gri görünür)
        disp.setTo(Scalar(255.0, 0.0, 0.0, 255.0), m1)         // kırmızı pikselleri kırmızı boya
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(disp, bmp)

        // Nozul çemberi (kalibrasyon + görsel); pahalı olduğu için 6 karede bir
        frameCount++
        if (frameCount % 6 == 0 || lastNozzle == null) {
            lastNozzle = detectNozzle(g, w, h)
        }
        val nozzle = lastNozzle

        // --- HEDEF MERKEZ: moda göre ---
        val manualX = refFracX.toDouble() * w
        val manualY = refFracY.toDouble() * h
        val nc = nozzle?.first
        val refX: Double
        val refY: Double
        when (centerMode) {
            CenterMode.AUTO -> {
                if (nc == null) {
                    return Result(false, w, h, null, 0.0, beam,
                        manualX, manualY, 0.0, 0.0, 0.0, 0.0, "Daire bulunamadı", State.NONE, bmp)
                }
                refX = nc.x; refY = nc.y
            }
            CenterMode.MANUAL -> { refX = manualX; refY = manualY }
            CenterMode.AUTO_THEN_MANUAL -> {
                refX = nc?.x ?: manualX
                refY = nc?.y ?: manualY
            }
        }

        if (beam == null) {
            return Result(false, w, h, nozzle?.first, nozzle?.second ?: 0.0,
                null, refX, refY, 0.0, 0.0, 0.0, 0.0, "Kırmızı ışık yok", State.NONE, bmp)
        }

        val dx = beam.x - refX
        val dy = beam.y - refY
        val offPx = hypot(dx, dy)
        val offMm = if (mmPerPx > 0) offPx * mmPerPx else 0.0
        val state = classify(offPx, offMm)
        val dir = directionText(dx, dy, state)

        return Result(
            ok = true, frameW = w, frameH = h,
            nozzleCenter = nozzle?.first, nozzleRadiusPx = nozzle?.second ?: 0.0,
            beamCenter = beam, refX = refX, refY = refY,
            offsetPx = offPx, offsetMm = offMm, dx = dx, dy = dy,
            direction = dir, state = state, display = bmp
        )
    }

    /** Lazer lekesi: büyük/düzensiz alanları (ten) eler; küçük, yuvarlak lekeyi seçer. */
    private fun bestLaserBlob(binMask: Mat): Point? {
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(binMask, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        var best: MatOfPoint? = null
        var bestScore = 0.0
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < 6.0 || area > maxBlobArea) continue        // çok küçük gürültü / çok büyük ten
            val center = Point(); val radius = FloatArray(1)
            val c2f = MatOfPoint2f(*c.toArray())
            Imgproc.minEnclosingCircle(c2f, center, radius)
            val circleArea = Math.PI * radius[0] * radius[0]
            val circularity = if (circleArea > 0) area / circleArea else 0.0   // 1'e yakın = yuvarlak
            if (circularity < 0.45) continue                     // düzensiz şekilleri ele
            // Yuvarlaklık ağırlıklı; eşitlikte daha büyük/parlak leke
            val score = circularity * (1.0 + area / maxBlobArea)
            if (score > bestScore) { bestScore = score; best = c }
        }
        return best?.let {
            val mm = Imgproc.moments(it)
            if (mm.m00 != 0.0) Point(mm.m10 / mm.m00, mm.m01 / mm.m00) else null
        }
    }

    /** Nozul çemberi: Hough, olmazsa dairemsi kontur (kalibrasyon için). */
    private fun detectNozzle(g: Mat, w: Int, h: Int): Pair<Point, Double>? {
        val blur = work ?: Mat().also { work = it }
        Imgproc.medianBlur(g, blur, 5)
        val circles = Mat()
        Imgproc.HoughCircles(
            blur, circles, Imgproc.HOUGH_GRADIENT, 1.0,
            (h / 8.0).coerceAtLeast(20.0),
            120.0, 30.0, minNozzleRadius, maxNozzleRadius
        )
        val cx = w / 2.0; val cy = h / 2.0
        var best: Pair<Point, Double>? = null
        if (circles.cols() > 0) {
            var bestDist = Double.MAX_VALUE
            for (i in 0 until circles.cols()) {
                val c = circles.get(0, i) ?: continue
                val p = Point(c[0], c[1]); val r = c[2]
                val d = hypot(p.x - cx, p.y - cy)
                if (d < bestDist) { bestDist = d; best = Pair(p, r) }
            }
        }
        circles.release()
        return best
    }

    private fun classify(offPx: Double, offMm: Double): State {
        return if (mmPerPx > 0) when {
            offMm <= greenThreshMm -> State.GREEN
            offMm <= yellowThreshMm -> State.YELLOW
            else -> State.RED
        } else when {
            offPx <= greenThreshPx -> State.GREEN
            offPx <= yellowThreshPx -> State.YELLOW
            else -> State.RED
        }
    }

    private fun directionText(dxIn: Double, dyIn: Double, state: State): String {
        if (state == State.GREEN) return "MERKEZDE ✓"
        val dx = if (invertX) -dxIn else dxIn
        val dy = if (invertY) -dyIn else dyIn
        val parts = ArrayList<String>()
        val dead = 2.0
        if (dx > dead) parts.add("Sağa") else if (dx < -dead) parts.add("Sola")
        if (dy > dead) parts.add("Aşağı") else if (dy < -dead) parts.add("Yukarı")
        return if (parts.isEmpty()) "Az bir tık" else parts.joinToString(" + ")
    }

    private fun ensure(m: Mat?, rows: Int, cols: Int, type: Int): Mat {
        if (m != null && m.rows() == rows && m.cols() == cols && m.type() == type) return m
        m?.release()
        return Mat(rows, cols, type)
    }

    fun release() {
        listOf(yuv, rgba, rgbaRot, gray, rgb, hsv, mask, mask2, display, work).forEach { it?.release() }
        yuv = null; rgba = null; rgbaRot = null; gray = null; rgb = null
        hsv = null; mask = null; mask2 = null; display = null; work = null
    }
}
