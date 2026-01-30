import React from 'react';

interface JerseyNumberInputProps {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}

const JerseyNumberInput: React.FC<JerseyNumberInputProps> = ({
  value,
  onChange,
  disabled = false,
}) => {
  return (
    <div className="w-full">
      <label
        htmlFor="jersey-number"
        className="block text-sm font-medium text-gray-700 mb-2"
      >
        Jersey Number
      </label>
      <input
        id="jersey-number"
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        placeholder="Enter jersey number (e.g., 7)"
        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
      />
    </div>
  );
};

export default JerseyNumberInput;
