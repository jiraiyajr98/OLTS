package com.jiraiya.olts;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions;
import com.google.firebase.ml.naturallanguage.FirebaseNaturalLanguage;
import com.google.firebase.ml.naturallanguage.translate.FirebaseTranslateLanguage;
import com.google.firebase.ml.naturallanguage.translate.FirebaseTranslator;
import com.google.firebase.ml.naturallanguage.translate.FirebaseTranslatorOptions;
import com.googlecode.tesseract.android.TessBaseAPI;
import com.karumi.dexter.Dexter;
import com.karumi.dexter.MultiplePermissionsReport;
import com.karumi.dexter.PermissionToken;
import com.karumi.dexter.listener.PermissionRequest;
import com.karumi.dexter.listener.multi.MultiplePermissionsListener;
import com.theartofdev.edmodo.cropper.CropImage;
import com.theartofdev.edmodo.cropper.CropImageView;

import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MainActivity extends AppCompatActivity {

    private boolean hasPermission = false;
    private ImageView cropImageView;

    private TessBaseAPI tessBaseAPI;
    public static final String TAG = "OLTS_CV";
    private Bitmap imageBitmap = null;
    private ProgressDialog progressDialog;
    private Bitmap originalImageBitmap = null;
    private static final int maxBeng = 3328;
    private static final int maxEng = 1881;
    private ArrayList<String> englishTokenList = new ArrayList<>();
    private ArrayList<String> bengaliTokenList = new ArrayList<>();
    private Interpreter tfLite;
    private TextView translationText;
    private FirebaseTranslatorOptions firebaseTranslatorOptions;
    private FirebaseTranslator englishBengaliTranslator;
    private boolean isModelDownloaded = false;
    private ExecutorService executor;
    private TextView ocrResult;


    static {

        if (OpenCVLoader.initDebug()) {
            Log.d(TAG, "Started Open CV");
        } else {
            Log.d(TAG, "Failed Load Open CV");
        }

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button cameraButton = findViewById(R.id.camera_button);
        cropImageView = findViewById(R.id.cropImageView);
        Button thresholdButton = findViewById(R.id.threshold_button);
        Button contoursButton = findViewById(R.id.contours_button);
        Button ocrButton = findViewById(R.id.ocr_button);
        translationText = findViewById(R.id.textView);
        ocrResult = findViewById(R.id.ocr_result);
        Button onlineTranslateButton = findViewById(R.id.translate_online_button);

        executor = Executors.newSingleThreadExecutor();

        requestPermission();
        loadJson();

        try {
            tfLite = new Interpreter(loadModelFile(this.getAssets(), "nmt_05_12_19_test_beng.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_SHORT).show();
        }

        progressDialog = new ProgressDialog(MainActivity.this);
        progressDialog.setMessage("Working on it...");
        progressDialog.setCancelable(false);

        cameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Toast.makeText(MainActivity.this, "has Per " + hasPermission, Toast.LENGTH_SHORT).show();

                if (hasPermission) {
                    CropImage.activity()
                            .setGuidelines(CropImageView.Guidelines.ON)
                            .start(MainActivity.this);
                } else {
                    requestPermission();
                }

            }
        });

        thresholdButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                setImageInOpenCV();
            }
        });

        contoursButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                drawCounters();
            }
        });

        ocrButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if (originalImageBitmap != null) {

                    progressDialog.setMessage("Please Wait");
                    progressDialog.show();

                    Thread th = new Thread(new Runnable() {
                        @Override
                        public void run() {

                            Future<String> stringFuture = executor.submit(getText(originalImageBitmap));

                            String result = null;

                            try {
                                result = stringFuture.get();
                                final String finalResult = result;

                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        ocrResult.setText(finalResult);
                                        progressDialog.dismiss();
                                    }
                                });


                            } catch (Exception e) {
                                e.printStackTrace();
                            }



                            if (result == null) {

                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        translationText.setText("OCR FAILED");
                                    }
                                });


                            } else {

                                String[] words = filterText(result).toLowerCase().split("\\s+");

                                int wordsLength = words.length;

                                float[][] inputArray = new float[1][8];

                                if (wordsLength > 8) {
                                    for (int i = 0; i < 8; i++)
                                        inputArray[0][i] = getTokenNumber(englishTokenList, words[i]);
                                } else {
                                    for (int i = 0; i < 8; i++) {
                                        if (i >= wordsLength)
                                            inputArray[0][i] = 0.0f;
                                        else
                                            inputArray[0][i] = getTokenNumber(englishTokenList, words[i]);
                                    }
                                }

                                final String res = runModel(inputArray, bengaliTokenList);

                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        translationText.setText(res);
                                    }
                                });



                            }

                        }
                    });

                    th.start();

                } else
                    Toast.makeText(MainActivity.this, "Bitmap Empty", Toast.LENGTH_SHORT).show();
            }
        });

        onlineTranslateButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (originalImageBitmap != null) {

                    progressDialog.setMessage("Please Wait");
                    progressDialog.show();

                    Thread th = new Thread(new Runnable() {
                        @Override
                        public void run() {

                            Future<String> stringFuture = executor.submit(getText(originalImageBitmap));

                            String result = null;

                            try {
                                result = stringFuture.get();

                                final String finalResult = result;
                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        ocrResult.setText(finalResult);
                                        progressDialog.dismiss();
                                    }
                                });


                            } catch (Exception e) {
                                e.printStackTrace();
                            }


                            if (result == null) {
                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        translationText.setText("OCR FAILED");
                                    }
                                });

                            }
                            else
                                doFirebaseTranslation(result);

                        }
                    });

                    th.start();


                } else
                    Toast.makeText(MainActivity.this, "Bitmap Empty", Toast.LENGTH_SHORT).show();
            }
        });


    }

    private void doFirebaseTranslation(String result)
    {


        setUpFirebaseTranslation();
        isDownloaded();
        if (isModelDownloaded) {
            englishBengaliTranslator.translate(result)
                    .addOnSuccessListener(
                            new OnSuccessListener<String>() {
                                @Override
                                public void onSuccess(@NonNull final String translatedText) {
                                    // Translation successful.
                                    runOnUiThread(new Runnable() {
                                        @Override
                                        public void run() {
                                            translationText.setText(translatedText);
                                        }
                                    });


                                }
                            })
                    .addOnFailureListener(
                            new OnFailureListener() {
                                @Override
                                public void onFailure(@NonNull Exception e) {

                                    runOnUiThread(new Runnable() {
                                        @Override
                                        public void run() {
                                            Toast.makeText(MainActivity.this, "Failed to translate", Toast.LENGTH_SHORT).show();
                                        }
                                    });


                                }
                            });
        } else {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(MainActivity.this, "Model not downloaded yet", Toast.LENGTH_SHORT).show();
                }
            });

        }
    }

    private void setUpFirebaseTranslation() {
        if (firebaseTranslatorOptions == null) {
            firebaseTranslatorOptions = new FirebaseTranslatorOptions.Builder()
                    .setSourceLanguage(FirebaseTranslateLanguage.EN)
                    .setTargetLanguage(FirebaseTranslateLanguage.BN)
                    .build();
        }

        if (englishBengaliTranslator == null) {
            englishBengaliTranslator = FirebaseNaturalLanguage.getInstance().getTranslator(firebaseTranslatorOptions);
        }
    }

    private void isDownloaded() {
        FirebaseModelDownloadConditions conditions = new FirebaseModelDownloadConditions.Builder()
                .requireWifi()
                .build();

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                progressDialog.setMessage("Getting Model....");
                progressDialog.show();
            }
        });



        englishBengaliTranslator.downloadModelIfNeeded(conditions)
                .addOnSuccessListener(
                        new OnSuccessListener<Void>() {
                            @Override
                            public void onSuccess(Void v) {
                                // Model downloaded successfully. Okay to start translating.
                                // (Set a flag, unhide the translation UI, etc.)

                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        Toast.makeText(MainActivity.this, "Model downloaded", Toast.LENGTH_SHORT).show();
                                        progressDialog.dismiss();
                                    }
                                });


                                isModelDownloaded = true;
                            }
                        })
                .addOnFailureListener(
                        new OnFailureListener() {
                            @Override
                            public void onFailure(@NonNull Exception e) {

                                runOnUiThread(new Runnable() {
                                    @Override
                                    public void run() {
                                        Toast.makeText(MainActivity.this, "Failed to Download", Toast.LENGTH_SHORT).show();
                                        progressDialog.dismiss();
                                    }
                                });


                                isModelDownloaded = false;
                            }
                        });
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);


        if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE) {
            CropImage.ActivityResult result = CropImage.getActivityResult(data);
            if (resultCode == RESULT_OK) {
                Uri resultUri = result.getUri();

                try {
                    imageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), resultUri);

                    ByteArrayOutputStream out = new ByteArrayOutputStream();
                    imageBitmap.compress(Bitmap.CompressFormat.JPEG, 15, out);
                    imageBitmap = BitmapFactory.decodeStream(new ByteArrayInputStream(out.toByteArray()));
                    originalImageBitmap = imageBitmap;

                } catch (IOException e) {
                    e.printStackTrace();
                }

                cropImageView.setImageBitmap(imageBitmap);


            } else if (resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE) {
                Exception error = result.getError();

                Toast.makeText(this, "Error Cropping", Toast.LENGTH_SHORT).show();

            }
        }
    }

    private String filterText(String input) {
        String withoutAccent = Normalizer.normalize(input, Normalizer.Form.NFD);
        return withoutAccent.replaceAll("[^a-zA-Z ]", "");

    }

    private void requestPermission() {
        Dexter.withContext(this)
                .withPermissions(
                        Manifest.permission.CAMERA,
                        Manifest.permission.READ_EXTERNAL_STORAGE,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE
                ).withListener(new MultiplePermissionsListener() {
            @Override
            public void onPermissionsChecked(MultiplePermissionsReport multiplePermissionsReport) {
                Toast.makeText(MainActivity.this, "Granted", Toast.LENGTH_SHORT).show();
                hasPermission = true;
            }

            @Override
            public void onPermissionRationaleShouldBeShown(List<PermissionRequest> list, PermissionToken permissionToken) {

                permissionToken.continuePermissionRequest();

            }
        }).check();
    }


    private void setImageInOpenCV() {

        if (imageBitmap == null) {
            Toast.makeText(this, "Image Bitmap Not Set", Toast.LENGTH_SHORT).show();
        } else {
            try {

//                Mat src = Imgcodecs.imread(imageFile.getAbsolutePath());
                Mat src = new Mat();
                Mat dst = new Mat();

                BitmapFactory.Options options = new BitmapFactory.Options();
                options.inDither = false;
                options.inSampleSize = 4;

                int width = imageBitmap.getWidth();
                int height = imageBitmap.getHeight();

                Bitmap convertedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);

                Utils.bitmapToMat(imageBitmap, src);

                Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
                Imgproc.threshold(src, dst, 120, 255, Imgproc.THRESH_BINARY);

                Utils.matToBitmap(dst, convertedBitmap);

                imageBitmap = convertedBitmap;

                cropImageView.setImageBitmap(convertedBitmap);

            } catch (Exception e) {
                e.printStackTrace();
            }

        }
    }

    private void drawCounters() {
        if (imageBitmap == null) {
            Toast.makeText(this, "Image Bitmap Not Set", Toast.LENGTH_SHORT).show();
        } else {

            progressDialog.setMessage("Please Wait...");
            progressDialog.show();
            Thread th = new Thread(new Runnable() {
                @Override
                public void run() {
                    Mat src = new Mat();
                    Utils.bitmapToMat(imageBitmap, src);
                    Mat gray = new Mat();
                    Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY);

                    Imgproc.Canny(gray, gray, 50, 200);
                    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
                    Mat hierarchy = new Mat();

                    Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
                    Log.d(TAG, "List Size " + contours.size());
                    for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
                        Imgproc.drawContours(src, contours, contourIdx, new Scalar(255, 0, 0), 10);
                    }

                    Utils.matToBitmap(src, imageBitmap);


                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            progressDialog.dismiss();
                            cropImageView.setImageBitmap(imageBitmap);
                        }
                    });

                }
            });

            th.start();


        }
    }


    private Callable<String> getText(final Bitmap bitmap) {

        //prepareTessData();

        return new Callable<String>() {
            @Override
            public String call() throws Exception {
                try {
                    tessBaseAPI = new TessBaseAPI();
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
                String dataPath = getExternalFilesDir("/").getPath() + "/";
                try {
                    tessBaseAPI.init(dataPath, "eng");
                    tessBaseAPI.setImage(bitmap);
                } catch (RuntimeException e) {
                    e.printStackTrace();

                }

                String retStr = null;
                try {
                    retStr = tessBaseAPI.getUTF8Text();
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
                tessBaseAPI.end();
                return retStr;
            }
        };


    }

    /**
     * @param list = All the List of words we have from the JSON
     * @param key  = The Word which we want to find
     * @return = Token number from JSON
     */

    private int getTokenNumber(ArrayList<String> list, String key) {

        if (list.contains(key)) {
            return list.indexOf(key) + 1;
        } else {
            return 0;
        }
    }

    /**
     * @param list = List  of all Words from the JSON File
     * @param key  = The Token Number
     * @return = The word which is in the JSON File
     */

    private String getWordFromToken(ArrayList<String> list, int key) {

        if (key == 0)
            return "";
        else
            return list.get(key - 1);

    }


    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename) throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * @param name = File Name
     * @return = JSON String
     */

    public String loadJSONFromAsset(String name) {
        String json = null;
        try {
            InputStream is = MainActivity.this.getAssets().open(name);
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            json = new String(buffer, "UTF-8");
        } catch (IOException ex) {
            ex.printStackTrace();
            return null;
        }
        return json;
    }

    /**
     * @param inputVal = input tokenize array
     * @param list     = ArrayList to which want to convert
     * @return = Converted String
     */

    private String runModel(float[][] inputVal, ArrayList<String> list) {

        float[][][] outputVal = new float[1][8][3329];

        tfLite.run(inputVal, outputVal);

        StringBuilder stringBuilder = new StringBuilder();

        for (float[][] floats : outputVal) {
            for (float[] aFloat : floats) {

                stringBuilder.append(getWordFromToken(list, argMax(aFloat)));
                stringBuilder.append(" ");
            }
        }
        return stringBuilder.toString();
    }

    private static int argMax(float[] floatArray) {

        float max = floatArray[0];
        int index = 0;

        for (int i = 0; i < floatArray.length; i++) {
            if (max < floatArray[i]) {
                max = floatArray[i];
                index = i;
            }
        }
        return index;
    }

    private void loadJson() {

        ProgressDialog pd = new ProgressDialog(MainActivity.this);

        pd.setMessage("Loading Data..");

        pd.show();

        JSONObject bengaliJsonObject = null;
        JSONObject englishJsonObject = null;
        try {
            bengaliJsonObject = new JSONObject(loadJSONFromAsset("word_dict_beng.json"));
            englishJsonObject = new JSONObject(loadJSONFromAsset("word_dict_eng.json"));
            for (int i = 1; i < maxBeng; i++)
                bengaliTokenList.add((String) bengaliJsonObject.get(String.valueOf(i)));

            for (int i = 1; i < maxEng; i++)
                englishTokenList.add((String) englishJsonObject.get(String.valueOf(i)));
        } catch (JSONException e) {
            e.printStackTrace();
        }

        pd.dismiss();

    }

}
