"use client";

import React, { useState, useRef } from "react";
import { Upload, Camera, AlertCircle, Loader, CheckCircle } from "lucide-react";
import { Client } from "@gradio/client";

export default function YOLODetectionApp() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [spaceName] = useState("knighttto/yolov8s-safety-detection");
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const [useCamera, setUseCamera] = useState(false);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setUseCamera(true);
        setError(null);
      }
    } catch (err) {
      setError("Camera access denied or not available");
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setUseCamera(false);
    }
  };

  const captureFromCamera = () => {
    if (videoRef.current) {
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext("2d").drawImage(videoRef.current, 0, 0);

      canvas.toBlob((blob) => {
        const file = new File([blob], "camera-capture.jpg", {
          type: "image/jpeg",
        });
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        stopCamera();
      });
    }
  };

  // Main detection function using Gradio Client
  const handleDetect = async () => {
    if (!selectedFile) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      console.log("Connecting to Gradio Space...");
      
      // Connect to your Gradio Space
      const client = await Client.connect(spaceName);
      console.log("Connected successfully!");

      // Convert file to blob if needed
      const imageBlob = selectedFile instanceof Blob 
        ? selectedFile 
        : await fetch(URL.createObjectURL(selectedFile)).then(r => r.blob());

      console.log("Sending image for prediction...");

      // Call the predict function
      const result = await client.predict("/predict_image", {
        image: imageBlob,
      });

      console.log("Result received:", result);

      // Extract the results
      // result.data is an array: [annotated_image, detections_text]
      const [annotatedImage, detectionsText] = result.data;

      setResults({
        annotated_image: annotatedImage,
        detections_text: detectionsText,
      });

      console.log("Detection complete!");
    } catch (err) {
      console.error("Detection error:", err);
      setError(
        `Failed to detect objects: ${err.message}\n\n` +
        `Troubleshooting:\n` +
        `• Check if the Gradio Space is running\n` +
        `• Space name: ${spaceName}\n` +
        `• Try opening https://huggingface.co/spaces/${spaceName}\n` +
        `• Make sure the Space has finished building`
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            YOLOv8s Safety Detection
          </h1>
          <p className="text-gray-600">
            Upload an image or use your camera to detect safety objects
          </p>
          <a
            href={`https://huggingface.co/spaces/${spaceName}`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block mt-2 text-sm text-indigo-600 hover:text-indigo-800 underline"
          >
            Open Gradio Space →
          </a>
        </div>

        {/* Status Badge */}
        <div className="bg-white rounded-lg shadow-md p-4 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium text-gray-700">
                Connected to:
              </span>
              <code className="ml-2 text-sm bg-gray-100 px-2 py-1 rounded">
                {spaceName}
              </code>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-600">Ready</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Panel - Upload/Camera */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">
              Input Source
            </h2>

            {/* Camera Section */}
            {!useCamera ? (
              <button
                onClick={startCamera}
                className="w-full mb-4 px-4 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center justify-center gap-2"
              >
                <Camera size={20} />
                Use Camera
              </button>
            ) : (
              <div className="mb-4">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  className="w-full rounded-lg mb-2 bg-black"
                />
                <div className="flex gap-2">
                  <button
                    onClick={captureFromCamera}
                    className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    Capture
                  </button>
                  <button
                    onClick={stopCamera}
                    className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            {/* Upload Section */}
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-indigo-500 transition-colors cursor-pointer"
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="mx-auto mb-4 text-gray-400" size={48} />
              <p className="text-gray-600 mb-2 font-medium">
                Drag & drop an image here
              </p>
              <p className="text-sm text-gray-500">
                or click to browse (JPG, PNG)
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>

            {/* Preview */}
            {previewUrl && (
              <div className="mt-4">
                <p className="text-sm text-gray-600 mb-2 font-medium">Preview:</p>
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="w-full rounded-lg shadow-md border border-gray-200"
                />
                <button
                  onClick={handleDetect}
                  disabled={loading}
                  className="w-full mt-4 px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2 font-medium"
                >
                  {loading ? (
                    <>
                      <Loader className="animate-spin" size={20} />
                      Detecting...
                    </>
                  ) : (
                    <>
                      <CheckCircle size={20} />
                      Detect Objects
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                <div className="flex items-start gap-2">
                  <AlertCircle
                    className="text-red-600 flex-shrink-0 mt-0.5"
                    size={20}
                  />
                  <div className="flex-1">
                    <p className="text-red-700 text-sm font-medium mb-1">
                      Detection Failed
                    </p>
                    <p className="text-red-600 text-xs whitespace-pre-line">
                      {error}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Right Panel - Results */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold mb-4 text-gray-800">
              Detection Results
            </h2>

            {!results ? (
              <div className="flex flex-col items-center justify-center h-96 text-gray-400">
                <Upload size={64} className="mb-4 opacity-50" />
                <p className="text-center">
                  Results will appear here after detection
                </p>
                <p className="text-sm text-center mt-2">
                  Upload an image to get started
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Success Badge */}
                <div className="bg-green-50 border border-green-200 rounded-lg p-3 flex items-center gap-2">
                  <CheckCircle className="text-green-600" size={20} />
                  <span className="text-green-700 font-medium">
                    Detection Complete!
                  </span>
                </div>

                {/* Annotated Image */}
                {results.annotated_image && (
                  <div>
                    <p className="text-sm text-gray-600 mb-2 font-medium">
                      Annotated Image:
                    </p>
                    <div className="rounded-lg overflow-hidden shadow-md border border-gray-200">
                      <img
                        src={results.annotated_image.url || results.annotated_image}
                        alt="Detected objects"
                        className="w-full"
                      />
                    </div>
                  </div>
                )}

                {/* Detections Text */}
                {results.detections_text && (
                  <div>
                    <p className="text-sm text-gray-600 mb-2 font-medium">
                      Detections:
                    </p>
                    <div className="bg-gray-50 border border-gray-200 p-4 rounded-lg">
                      <div className="prose prose-sm max-w-none">
                        {results.detections_text.split('\n').map((line, i) => (
                          <p key={i} className="text-gray-800 my-1">
                            {line}
                          </p>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>
            Powered by YOLOv8s • Deployed on Hugging Face Spaces
          </p>
        </div>
      </div>
    </div>
  );
}