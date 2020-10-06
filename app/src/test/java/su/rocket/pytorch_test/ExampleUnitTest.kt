package su.rocket.pytorch_test

import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.opencv.core.Mat
import org.opencv.videoio.VideoCapture
import java.io.File
import java.util.ArrayList


/**
 * Example local unit test, which will execute on the development machine (host).
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
class ExampleUnitTest {
    @Test
    fun addition_isCorrect() {
        val file = File("src/test/resources/input.mp4")
        val absolutePath: String = file.getAbsolutePath()

        val videoCapture = VideoCapture()
        videoCapture.open(absolutePath)
        val clip = ArrayList<Mat>()
        for (i in 0..63) {
            val mat = Mat()
            videoCapture.read(mat)
            clip.add(mat)
        }

        ClipUtils.prepare(clip);

        assertEquals(1, 0)
    }
}