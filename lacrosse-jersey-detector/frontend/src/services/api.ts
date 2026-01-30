import axios from 'axios';
import type {
  UploadResponse,
  AnalyzeRequest,
  AnalyzeResponse,
  AnalysisResult,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const uploadVideo = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<UploadResponse>('/api/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const analyzeVideo = async (
  request: AnalyzeRequest
): Promise<AnalyzeResponse> => {
  const response = await api.post<AnalyzeResponse>('/api/analyze', request);
  return response.data;
};

export const getResults = async (jobId: string): Promise<AnalysisResult> => {
  const response = await api.get<AnalysisResult>(`/api/results/${jobId}`);
  return response.data;
};

export default api;
