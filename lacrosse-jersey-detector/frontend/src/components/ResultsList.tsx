import React from 'react';
import type { TimestampInterval } from '../types';

interface ResultsListProps {
  intervals: TimestampInterval[];
  jerseyNumber: string;
  isLoading?: boolean;
}

const ResultsList: React.FC<ResultsListProps> = ({
  intervals,
  jerseyNumber,
  isLoading = false,
}) => {
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (isLoading) {
    return (
      <div className="text-center text-gray-500 py-8">
        Processing video... This may take a few minutes.
      </div>
    );
  }

  if (intervals.length === 0) {
    return (
      <div className="text-center text-gray-500 py-8">
        No detections found for jersey number {jerseyNumber}.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">
        Found {intervals.length} play{intervals.length !== 1 ? 's' : ''} for
        jersey #{jerseyNumber}
      </h3>
      <div className="space-y-2">
        {intervals.map((interval, index) => (
          <div
            key={index}
            className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
          >
            <div className="flex justify-between items-center">
              <div>
                <span className="text-sm font-medium text-gray-600">
                  Play #{index + 1}
                </span>
                <p className="text-lg font-semibold text-gray-900">
                  {formatTime(interval.start_time)} -{' '}
                  {formatTime(interval.end_time)}
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm text-gray-500">
                  Duration: {formatTime(interval.duration)}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ResultsList;
