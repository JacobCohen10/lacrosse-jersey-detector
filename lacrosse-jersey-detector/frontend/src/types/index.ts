export interface UploadResponse {
  video_id: string;
  filename: string;
  message: string;
}

export interface AnalyzeRequest {
  video_id: string;
  jersey_number: string;
  fast_mode?: boolean;
}

export interface AnalyzeResponse {
  job_id: string;
  message: string;
  status: string;
}

export interface TimestampInterval {
  start_time: number;
  end_time: number;
  duration: number;
}

export interface AnalysisResult {
  job_id: string;
  status: 'processing' | 'completed' | 'failed';
  jersey_number: string;
  video_id: string;
  intervals: TimestampInterval[];
  total_detections: number;
  created_at?: string;
  completed_at?: string;
  error?: string;
}
