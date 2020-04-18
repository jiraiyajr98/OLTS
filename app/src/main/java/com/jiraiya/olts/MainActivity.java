package com.jiraiya.olts;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Toast;

import com.karumi.dexter.Dexter;
import com.karumi.dexter.PermissionToken;
import com.karumi.dexter.listener.PermissionDeniedResponse;
import com.karumi.dexter.listener.PermissionGrantedResponse;
import com.karumi.dexter.listener.PermissionRequest;
import com.karumi.dexter.listener.single.DialogOnDeniedPermissionListener;
import com.karumi.dexter.listener.single.PermissionListener;
import com.theartofdev.edmodo.cropper.CropImage;
import com.theartofdev.edmodo.cropper.CropImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private boolean hasPermission = false;
    private ImageView cropImageView;

    public static final String TAG = "OLTS_CV";
    private Bitmap imageBitmap = null;
    ProgressDialog progressDialog ;

    static{

        if(OpenCVLoader.initDebug())
        {
            Log.d(TAG,"Started Open CV");
        }
        else
        {
            Log.d(TAG,"Failed Load Open CV");
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


        requestPermission();

        progressDialog = new ProgressDialog(MainActivity.this);
        progressDialog.setMessage("Working on it...");
        progressDialog.setCancelable(false);

        cameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Toast.makeText(MainActivity.this, "has Per "+hasPermission, Toast.LENGTH_SHORT).show();

                if(hasPermission)
                {
                    CropImage.activity()
                            .setGuidelines(CropImageView.Guidelines.ON)
                            .start(MainActivity.this);
                }
                else
                {
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

    private void requestPermission()
    {
        Dexter.withContext(this)
                .withPermission(Manifest.permission.CAMERA)
                .withListener(new PermissionListener() {
                    @Override
                    public void onPermissionGranted(PermissionGrantedResponse response)
                    {
                        hasPermission = true;
                    }
                    @Override
                    public void onPermissionDenied(PermissionDeniedResponse response)
                    {
                        hasPermission = false;
                    }
                    @Override
                    public void onPermissionRationaleShouldBeShown(PermissionRequest permission, PermissionToken token)
                    {
                        token.continuePermissionRequest();
                    }
                }).check();
    }


    private void setImageInOpenCV()
    {

        if(imageBitmap == null )
        {
            Toast.makeText(this, "Image Bitmap Not Set", Toast.LENGTH_SHORT).show();
        }
        else {
            try {

//                Mat src = Imgcodecs.imread(imageFile.getAbsolutePath());
                Mat src = new Mat();
                Mat dst = new Mat();

                BitmapFactory.Options options = new BitmapFactory.Options();
                options.inDither = false;
                options.inSampleSize = 4;

                int width = imageBitmap.getWidth();
                int height = imageBitmap.getHeight();

                Bitmap convertedBitmap = Bitmap.createBitmap(width,height,Bitmap.Config.RGB_565);

                Utils.bitmapToMat(imageBitmap,src);

                Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
                Imgproc.threshold(src, dst, 120, 255,Imgproc.THRESH_BINARY);

                Utils.matToBitmap(dst,convertedBitmap);

                imageBitmap = convertedBitmap;

                cropImageView.setImageBitmap(convertedBitmap);

            }
            catch (Exception e)
            {
                e.printStackTrace();
            }

        }
    }

    private void drawCounters()
    {
        if(imageBitmap == null)
        {
            Toast.makeText(this, "Image Bitmap Not Set", Toast.LENGTH_SHORT).show();
        }
        else
        {

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

                    Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_TREE,Imgproc.CHAIN_APPROX_SIMPLE);
                    Log.d(TAG,"List Size "+contours.size());
                    for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
                        Imgproc.drawContours(src, contours, contourIdx, new Scalar(255, 0, 0), 10);
                    }

                    Utils.matToBitmap(src,imageBitmap);


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



}
