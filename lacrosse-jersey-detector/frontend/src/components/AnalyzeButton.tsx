import React from 'react';
import LoadingSpinner from './LoadingSpinner';

interface AnalyzeButtonProps {
  onClick: () => void;
  disabled?: boolean;
  loading?: boolean;
}

const AnalyzeButton: React.FC<AnalyzeButtonProps> = ({
  onClick,
  disabled = false,
  loading = false,
}) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className={`w-full py-3 px-6 rounded-lg font-semibold text-white transition-colors ${
        disabled || loading
          ? 'bg-gray-400 cursor-not-allowed'
          : 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800'
      }`}
    >
      {loading ? (
        <span className="flex items-center justify-center gap-2">
          <LoadingSpinner size="sm" />
          Analyzing...
        </span>
      ) : (
        'Analyze'
      )}
    </button>
  );
};

export default AnalyzeButton;
