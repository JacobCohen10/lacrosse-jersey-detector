import React, { useState, useEffect } from 'react';
import VideoUpload from './components/VideoUpload';
import JerseyNumberInput from './components/JerseyNumberInput';
import AnalyzeButton from './components/AnalyzeButton';
import ResultsList from './components/ResultsList';
import { uploadVideo, analyzeVideo, getResults } from './services/api';
import type { TimestampInterval } from './types';

const App: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [jerseyNumber, setJerseyNumber] = useState<string>('');
  const [fastMode, setFastMode] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [intervals, setIntervals] = useState<TimestampInterval[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(false);

  // Poll for results when analysis is in progress
  useEffect(() => {
    if (!jobId || !isPolling) return;

    const pollInterval = setInterval(async () => {
      try {
        const result = await getResults(jobId);
        
        if (result.status === 'completed') {
          setIntervals(result.intervals);
          setIsPolling(false);
          setIsAnalyzing(false);
        } else if (result.status === 'failed') {
          setError(result.error || 'Analysis failed');
          setIsPolling(false);
          setIsAnalyzing(false);
        }
        // If still processing, continue polling
      } catch (err) {
        console.error('Error polling results:', err);
        setError('Failed to fetch results');
        setIsPolling(false);
        setIsAnalyzing(false);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(pollInterval);
  }, [jobId, isPolling]);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setVideoId(null);
    setIntervals([]);
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a video file');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const response = await uploadVideo(selectedFile);
      setVideoId(response.video_id);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to upload video');
    } finally {
      setIsUploading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!videoId) {
      setError('Please upload a video first');
      return;
    }

    if (!jerseyNumber.trim()) {
      setError('Please enter a jersey number');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setIntervals([]);

    try {
      const response = await analyzeVideo({
        video_id: videoId,
        jersey_number: jerseyNumber.trim(),
        fast_mode: fastMode,
      });
      setJobId(response.job_id);
      setIsPolling(true);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start analysis');
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Lacrosse Jersey Detector
          </h1>
          <p className="text-gray-600">
            Upload game footage and find plays by jersey number
          </p>
        </header>

        <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
          {/* Video Upload Section */}
          <section>
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Step 1: Upload Video
            </h2>
            <VideoUpload
              onFileSelect={handleFileSelect}
              selectedFile={selectedFile}
              disabled={isUploading || isAnalyzing}
            />
            {!videoId && selectedFile && (
              <button
                onClick={handleUpload}
                disabled={isUploading}
                className="mt-4 w-full py-2 px-4 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {isUploading ? 'Uploading...' : 'Upload Video'}
              </button>
            )}
            {videoId && (
              <p className="mt-2 text-sm text-green-600">
                ✓ Video uploaded successfully
              </p>
            )}
          </section>

          {/* Jersey Number Input Section */}
          <section>
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Step 2: Enter Jersey Number
            </h2>
            <JerseyNumberInput
              value={jerseyNumber}
              onChange={setJerseyNumber}
              disabled={!videoId || isAnalyzing}
            />
          </section>

          {/* Analyze Button Section */}
          <section>
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Step 3: Analyze
            </h2>
            <label className="flex items-center gap-2 mb-4 cursor-pointer">
              <input
                type="checkbox"
                checked={fastMode}
                onChange={(e) => setFastMode(e.target.checked)}
                disabled={isAnalyzing}
                className="rounded border-gray-300 text-green-600 focus:ring-green-500"
              />
              <span className="text-sm text-gray-700">
                Fast mode (fewer frames &amp; strategies — 5–10× faster, slightly lower recall)
              </span>
            </label>
            <AnalyzeButton
              onClick={handleAnalyze}
              disabled={!videoId || !jerseyNumber.trim()}
              loading={isAnalyzing}
            />
          </section>

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-red-800">{error}</p>
            </div>
          )}

          {/* Results Section */}
          {(intervals.length > 0 || isAnalyzing) && (
            <section>
              <h2 className="text-xl font-semibold text-gray-800 mb-4">
                Results
              </h2>
              <ResultsList
                intervals={intervals}
                jerseyNumber={jerseyNumber}
                isLoading={isAnalyzing}
              />
            </section>
          )}

          {/* No results message after completed analysis */}
          {!isAnalyzing && !isPolling && jobId && intervals.length === 0 && !error && (
            <section>
              <h2 className="text-xl font-semibold text-gray-800 mb-2">
                Results
              </h2>
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <p className="text-yellow-800">
                  No clips were found for jersey number <span className="font-semibold">{jerseyNumber}</span>.
                </p>
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
