
package com.example.user.myapplication;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;


public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "MainActivity";

    private CameraBridgeViewBase openCvCameraView;
    private CascadeClassifier cascadeClassifier;
    private Mat grayscaleImage;
    private int absoluteFaceSize;
    private int averageframes=5;
    private int frameCounter=0;
    List<MatOfRect> averageFaces = new ArrayList<MatOfRect>();

    static {
        System.loadLibrary("tensorflow_inference");
    }

    //private static final String MODEL_FILE = "file:///android_asset/frozen_age_graph.pb";
    //private static final String MODEL_LABELS = "file:///android_asset/ages.txt";
    //private static final String INPUT_NODE = "batch_processing/Reshape";
    //private static final String OUTPUT_NODE = "output/output";

    private static final String MODEL_FILE = "file:///android_asset/frozen_age_graph.pb";
    private static final String MODEL_LABELS = "file:///android_asset/age_labels.txt";
//    private static final String INPUT_NODE = "Placeholder:0";
//    private static final String INPUT_NODE = "batch_processing/Reshape";

    private static final String INPUT_NODE = "input";
    private static final String OUTPUT_NODE = "output/output";

    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;

    private static final int MAX_FACES=6;
    private static final int texBackgroundWaiting = Color.parseColor("#ccff0000");
    private static final int texBackground = Color.parseColor("#ccffffff");

    //MAX_BATCH_SZ = 128

    //AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
    private Vector<String> labels = new Vector<String>();
    //RESIZE_FINAL = 227
    private static final int IMAGE_INPUT_SIZE = 227;
    //Color Image? or grayscale and normalized?
    private static final int[] INPUT_SIZE = {IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3};




    private TensorFlowInferenceInterface inferenceInterface;

    private static boolean GET_AGE = false;
    private final BlockingQueue<FaceProcessor> faceProcessQueue = new ArrayBlockingQueue<>(100);

    private Thread mProcessThread = null;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("OpenCV", "OpenCV loaded successfully");
                    mOpenCvCameraView.setCvCameraViewListener(MainActivity.this);
                    load_cascade();
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    private CameraBridgeViewBase mOpenCvCameraView;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_camera_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        String actualFilename = MODEL_LABELS.split("file:///android_asset/")[1];
        Log.i(TAG, "Reading labels from: " + actualFilename);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(getAssets().open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!" , e);
        }

        Log.i("OnCreate", "Labels:" + labels.toArray().toString());
        String modelActualFilename = MODEL_FILE.split("file:///android_asset/")[1];
        Log.i("onCreate", "Reading model from: " + modelActualFilename);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        final Operation operation = inferenceInterface.graphOperation(OUTPUT_NODE);
        final int numClasses = (int) operation.output(0).shape().size(1);
        Log.i(TAG, "Read " + labels.size() + " labels, output layer size is " + numClasses);

    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

        if (mProcessThread != null)
            mProcessThread.interrupt();

        // Reset queue
        faceProcessQueue.clear();
    }
    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

        FaceConsumer consumer = new FaceConsumer(faceProcessQueue);
        mProcessThread = new Thread(consumer);
        mProcessThread.start();
        System.out.println("faceConsumer has been started");
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private void load_cascade() {

        // Copy the resource into a temp file so OpenCV can load it
        try (InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface)) {
            File cascadeDir = getDir("cascades", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            try (FileOutputStream os = new FileOutputStream(mCascadeFile)) {

                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }

                // Load the cascade classifier
                cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                if (!cascadeClassifier.load(mCascadeFile.getAbsolutePath())) {
                    Log.e("OpenCVActivity", "Failed to load cascade classifier");
                } else {
                    Log.i("OpenCVActivity", "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
                }
            }
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
            throw new RuntimeException("Error loading cascade resources", e);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        // FIXME:Don't like this... I get it... don't need to initialize a variable each time, but..
        // maybe create an image object with syncronization for thread safety...
        grayscaleImage = new Mat(height, width, CvType.CV_8UC4);

        // The faces will be a 20% of the height of the screen
        // TODO: resize for each face
        absoluteFaceSize = (int) (height * 0.2);
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame aInputFrame) {
        // Create a grayscale image
        Mat colorImage = aInputFrame.rgba();
        //Mat greyImage = aInputFrame.gray();
        Imgproc.cvtColor(colorImage, grayscaleImage, Imgproc.COLOR_RGBA2RGB);

        MatOfRect faces = new MatOfRect();

        // Use the classifier to detect faces
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(grayscaleImage, faces, 1.1, 5, 2,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }



        //MatOfRect averagedFaces = getAverageFaces(faces);

        final int[] faceViews = new int[]{R.id.face1, R.id.face2, R.id.face3, R.id.face4, R.id.face5, R.id.face6};
        final int[] ageViews = new int[]{R.id.age1, R.id.age2, R.id.age3, R.id.age4, R.id.age5, R.id.age6};

        // If there are any faces found, draw a rectangle around it
        Rect[] facesArray = faces.toArray();
        final List<Bitmap> faceBmps = new ArrayList<Bitmap>();
        //final List<Bitmap> greyFaceBmps = new ArrayList<Bitmap>();

        Bitmap frameBmp = Bitmap.createBitmap(colorImage.width(), colorImage.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(colorImage, frameBmp);
        for (int i = 0; i < facesArray.length; i++) {
            Rect face = facesArray[i];
            Imgproc.rectangle(colorImage, face.tl(), face.br(), new Scalar(0, 255, 0, 255), 3);
            Bitmap faceBmp = Bitmap.createBitmap(frameBmp, (int) face.tl().x, (int) face.tl().y, face.width, face.height);
            faceBmps.add(faceBmp);

            //Bitmap greyFaceBmp = Bitmap.createBitmap(frameBmp, (int) face.tl().x, (int) face.tl().y, face.width, face.height);
            //greyFaceBmps.add(faceBmp);
        }

        if (GET_AGE) {
            GET_AGE = false;
            FaceProcessor ageProcessor = new FaceProcessor() {
                @Override
                public void processFace() {
                    String[] outputNames = new String[] {OUTPUT_NODE};
                    int[] intValues = new int[IMAGE_INPUT_SIZE * IMAGE_INPUT_SIZE];
                    float[] floatValues = new float[IMAGE_INPUT_SIZE * IMAGE_INPUT_SIZE * 3];
                    float[] outputs = new float[8];
                    //final List<String> ages = new ArrayList<>();

                    //Initailize to 0
                    for (int idx = 0; idx < outputs.length; idx++) {
                        outputs[idx] = 0;
                    }

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            // Set face and age as invisible
                            for (int idx = 0; idx < faceViews.length; idx++) {
                                int faceviewId = faceViews[idx];
                                int ageviewId =  ageViews[idx];
                                ImageView faceview = (ImageView) findViewById(faceviewId);
                                TextView ageview = (TextView) findViewById(ageviewId);

                                faceview.setVisibility(View.INVISIBLE);
                                ageview.setVisibility(View.INVISIBLE);

                                // For at most max faces detected, display each face
                                if ( faceBmps.size() > idx && idx < MAX_FACES ) {
                                    faceview.setImageBitmap(resize(faceBmps.get(idx), 50, 50));

                                    ageview.setText("--");
                                    ageview.setBackgroundColor(texBackgroundWaiting);
                                    faceview.setVisibility(View.VISIBLE);
                                    ageview.setVisibility(View.VISIBLE);
                                }
                            }
                        }
                    });

                    Log.w("processFace", "Processing Faces!");

                    for (int idx = 0; idx < faceBmps.size(); idx++) {

                        if (idx >= MAX_FACES) {
                            break;
                        }

                        int ageviewId =  ageViews[idx];
                        final TextView ageview = (TextView) findViewById(ageviewId);

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                // Update Age field with processing indicator
                                ageview.setText("...");
                                ageview.setBackgroundColor(texBackground);
                            }
                        });

                        Bitmap faceBmp = faceBmps.get(idx);

                        //TODO: Convert to greyscale?
                        //Mat mat = new Mat();
                        //Bitmap bmp32 = faceBmp.copy(Bitmap.Config.ARGB_8888, true);
                        //Utils.bitmapToMat(bmp32, mat);

                        //Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY);
                        //Utils.matToBitmap(mat, faceBmp);
                        Bitmap resizedFace = resize(faceBmp, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE);

//                        ByteArrayOutputStream stream = new ByteArrayOutputStream();
//                        faceBmp.compress(Bitmap.CompressFormat.JPEG, 100, stream);
//                        byte[] byteArray = stream.toByteArray();
//                        ByteBuffer byteBuffer = ByteBuffer.wrap(byteArray);
//                        String byteString = stream.toString();

                        // Log this method so that it can be analyzed with systrace.

                        // Preprocess the image data from 0-255 int to normalized float based
                        // on the provided parameters.

                        Log.i("processFace", "Normalizing Input Image:" + resizedFace.getWidth() + ", " + resizedFace.getHeight());
                        resizedFace.getPixels(intValues, 0, resizedFace.getWidth(), 0, 0, resizedFace.getWidth(), resizedFace.getHeight());
                        for (int i = 0; i < intValues.length; ++i) {
                            final int val = intValues[i];
                            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
                            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
                            floatValues[i * 3 + 2] = ((val & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
                        }


                        // Get age from TF model
                        // Copy the input data into TensorFlow.
                        Log.i("processFace", "Feeding Input to model");
                        inferenceInterface.feed(INPUT_NODE, floatValues, 1, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3);

                        // Run the inference call.
                        Log.i("processFace", "Running Model");
                        inferenceInterface.run(outputNames, false);

                        // Copy the output Tensor back into the output array.
                        Log.i("processFace", "Fetching Output of Model");
                        inferenceInterface.fetch(OUTPUT_NODE, outputs);

                        Log.i("processFace", "Output Of Model:"+ Arrays.toString(outputs));

                        // Find Age
                        int max_idx = 0;
                        for (int idx_out = 0; idx_out < outputs.length; idx_out++) {
                            if ( outputs[idx_out] > outputs[max_idx] ) {
                                max_idx = idx_out;
                            }
                        }

                        final String age = labels.get(max_idx);

                        Log.i("processFace", "Estimated Age("+max_idx+"):"+ labels.get(max_idx));
                        //ages.add(labels.get(max_idx));

                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                // Update Age field with age
                                Log.i("FaceGui", "Estimated Age:"+ age);
                                ageview.setText(age);
                            }
                        });
                    }
                }
            };

            try {
                faceProcessQueue.put(ageProcessor);
            } catch ( InterruptedException ex ) {
                Log.e("OnCameraFrame", "Thread for face Processing has stopped!");
            }
        }

        return colorImage;
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if ( event.getAction() == MotionEvent.ACTION_UP) {
            GET_AGE=true;
        }
        return super.onTouchEvent(event);
    }

    private static Bitmap resize(Bitmap image, int maxWidth, int maxHeight) {
        if (maxHeight > 0 && maxWidth > 0) {
            int width = image.getWidth();
            int height = image.getHeight();
            float ratioBitmap = (float) width / (float) height;
            float ratioMax = (float) maxWidth / (float) maxHeight;

            int finalWidth = maxWidth;
            int finalHeight = maxHeight;
            if (ratioMax > ratioBitmap) {
                finalWidth = (int) ((float)maxHeight * ratioBitmap);
            } else {
                finalHeight = (int) ((float)maxWidth / ratioBitmap);
            }
            image = Bitmap.createScaledBitmap(image, finalWidth, finalHeight, true);
            return image;
        } else {
            return image;
        }
    }

    private interface FaceProcessor {
        void processFace();
    }

    class FaceConsumer implements Runnable {
        private final BlockingQueue<FaceProcessor> queue;
        FaceConsumer(BlockingQueue q) { queue = q; }
        public void run() {
            try {
                while (true) { consume(queue.take()); }
            } catch (InterruptedException ex) {
                Log.i("FaceConsumer", "Face Processing has stopped");
                return;
            }
        }

        void consume(FaceProcessor aFaceProcessor) {
            aFaceProcessor.processFace();
        }
    }


}