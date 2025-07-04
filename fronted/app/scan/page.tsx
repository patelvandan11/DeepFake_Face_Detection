'use client';

import { useRef, useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

const videoConstraints = {
  width: 640,
  height: 480,
  facingMode: 'user',
};

const ScanPage = () => {
  const webcamRef = useRef<Webcam>(null);
  const [source, setSource] = useState('custom');
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<{ isFake: boolean; confidence: number } | null>(null);

  const handleScan = async () => {
    if (!webcamRef.current) return;

    const screenshot = webcamRef.current.getScreenshot();
    if (!screenshot) return alert('Failed to capture image from webcam.');

    setScanning(true);
    setResult(null);

    try {
      const response = await axios.post('/api/detect', {
        image: screenshot,
        source,
      });

      setResult(response.data);
    } catch (error) {
      console.error('Detection failed', error);
      alert('Detection failed. Please try again.');
    } finally {
      setScanning(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col items-center justify-center px-4 py-8">
      <div className="w-full max-w-xl">
        <h1 className="text-4xl font-bold text-center mb-6">🎯 Live Exam Proctoring</h1>

        <div className="bg-gray-900 p-6 rounded-xl shadow-xl">
          <div className="mb-4">
            <label className="mr-2 font-medium text-gray-300">Select Source:</label>
            <select
              value={source}
              onChange={(e) => setSource(e.target.value)}
              className="bg-gray-800 text-white border border-gray-600 rounded px-4 py-2 w-full"
            >
              <option value="zoom">Zoom</option>
              <option value="google_meet">Google Meet</option>
              <option value="custom">Custom Software</option>
            </select>
          </div>

          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={videoConstraints}
            className="rounded-lg shadow-lg w-full mb-4 border border-gray-700"
          />

          <button
            onClick={handleScan}
            disabled={scanning}
            className={`w-full py-2 rounded-lg font-semibold transition duration-300 
              ${scanning
                ? 'bg-gray-700 cursor-not-allowed text-gray-300'
                : 'bg-blue-600 hover:bg-blue-700 text-white'}`}
          >
            {scanning ? (
              <span className="flex items-center justify-center">
                <svg
                  className="animate-spin h-5 w-5 mr-2 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                  />
                </svg>
                Scanning...
              </span>
            ) : (
              'Start Scan'
            )}
          </button>
        </div>

        {result && (
          <div className="mt-8 bg-gray-900 rounded-xl p-6 shadow-xl">
            <h2 className="text-2xl font-semibold mb-4 text-center">🧠 Detection Result</h2>
            {result.isFake ? (
              <div className="flex items-center justify-center text-red-400 text-lg font-medium">
                <svg
                  className="h-6 w-6 mr-2"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2}
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 9v2m0 4h.01M4.93 19h14.14c.87 0 1.38-.98.94-1.7L13.94 5.8a1 1 0 00-1.73 0L4 17.3c-.44.72.07 1.7.93 1.7z"
                  />
                </svg>
                Fake Detected – Confidence: {result.confidence.toFixed(2)}%
              </div>
            ) : (
              <div className="flex items-center justify-center text-green-400 text-lg font-medium">
                <svg
                  className="h-6 w-6 mr-2"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2}
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M5 13l4 4L19 7"
                  />
                </svg>
                Genuine – Confidence: {result.confidence.toFixed(2)}%
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ScanPage;