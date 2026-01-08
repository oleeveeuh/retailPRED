import React from 'react';
import CounterfactualExplorer from '../components/CounterfactualExplorer';
import { useNavigate } from 'react-router-dom';

const CounterfactualPage: React.FC = () => {
  const navigate = useNavigate();

  const handleApplyScenario = (scenario: any) => {
    console.log('Applying scenario:', scenario);
    // Navigate back to dashboard with scenario applied
    navigate('/');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      <div className="container mx-auto">
        <CounterfactualExplorer
          predictionId={1}
          currentPrediction={42500}
          onApplyScenario={handleApplyScenario}
        />
      </div>
    </div>
  );
};

export default CounterfactualPage;
