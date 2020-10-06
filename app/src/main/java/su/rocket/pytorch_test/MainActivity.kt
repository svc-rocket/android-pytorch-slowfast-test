package su.rocket.pytorch_test

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.AsyncTask
import android.os.Bundle
import android.text.TextUtils
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import su.rocket.pytorch_test.Utils.assetFilePath
import java.lang.StringBuilder
import java.util.*


class MainActivity : Activity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private var time: Date = Date()
    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    private var mStatusView: TextView? = null
    private var module: Module? = null
    private val matQueue = mutableListOf<Mat>()

    var mRgba: Mat? = null
    var mRgbaF: Mat? = null
    var mRgbaT: Mat? = null

    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    mOpenCvCameraView!!.enableView()
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        Log.i(TAG, "called onCreate")
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        module = Module.load(assetFilePath(this, "optimized_scripted.pt"))

        // Permissions for Android 6+
        ActivityCompat.requestPermissions(
            this@MainActivity,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_REQUEST
        )

        setContentView(R.layout.activity_main)

        mStatusView = findViewById(R.id.status)
        mOpenCvCameraView = findViewById(R.id.main_surface)
        mOpenCvCameraView!!.setMaxFrameSize(350, 350)
        mOpenCvCameraView!!.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView!!.setCvCameraViewListener(this)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        when (requestCode) {
            CAMERA_PERMISSION_REQUEST -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    mOpenCvCameraView!!.setCameraPermissionGranted()
                } else {
                    val message = "Camera permission was not granted"
                    Log.e(TAG, message)
                    Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                }
            }
            else -> {
                Log.e(TAG, "Unexpected permission request")
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if (mOpenCvCameraView != null)
            mOpenCvCameraView!!.disableView()
    }

    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (mOpenCvCameraView != null)
            mOpenCvCameraView!!.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        mRgba = Mat(height, width, CvType.CV_8UC4)
        mRgbaF = Mat(height, width, CvType.CV_8UC4)
        mRgbaT = Mat(width, width, CvType.CV_8UC4)
    }

    override fun onCameraViewStopped() {
        mRgba?.release()
    }

    override fun onCameraFrame(frame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        mRgba = frame.rgba()

        // Rotate mRgba 90 degrees
        Core.transpose(mRgba, mRgbaT);
        Imgproc.resize(mRgbaT, mRgbaF, mRgbaF!!.size(), 0.0, 0.0, 0);
        Core.flip(mRgbaF, mRgba, 1);

        if (module != null) {
            if (matQueue.size < 64) {
                setStatus("Collecting...");
                matQueue.add(mRgba!!.clone())

                if (matQueue.size == 64) {
                    time = Date()
                    InferenceTask().execute()
                }
            } else {
                setStatus(String.format("Inference %dms", ((Date().time - time.time))));
            }
        }

        return mRgba!!
    }

    fun showInferenceResult(result: String) {
        runOnUiThread { Toast.makeText(baseContext, result, Toast.LENGTH_LONG).show() }
    }

    fun setStatus(status: String) {
        runOnUiThread { mStatusView?.setText(status) }
    }

    inner class InferenceTask : AsyncTask<Void, Void, FloatArray>(){
        override fun doInBackground(vararg params: Void): FloatArray {
            val tensors = ClipUtils.prepare(matQueue)

            val outputTensor: Tensor = module?.forward(
                IValue.listFrom(tensors[0], tensors[1])
            )!!.toTensor()
            val scores: FloatArray = outputTensor.getDataAsFloatArray()
            System.gc()
            return scores
        }

        override fun onPreExecute() {
            super.onPreExecute()

        }

        override fun onPostExecute(scores: FloatArray) {
            matQueue.clear()
            val indexes = Utils.topK(scores, 5)

            val labelsBuilder = StringBuilder()
            for (index in indexes) {
                if (!labelsBuilder.isEmpty()) labelsBuilder.append(", ")
                labelsBuilder.append(KinetikUtils.map.get(index))
            }

            if (!labelsBuilder.isEmpty()) showInferenceResult(labelsBuilder.toString())
        }
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val CAMERA_PERMISSION_REQUEST = 1
    }
}