'use client';

import dynamic from 'next/dynamic';

// Correct dynamic imports for named exports
const FashionPredictionForm = dynamic(
  () => import('@/components/fashion-prediction-form'),
  { ssr: false }
);

const FashionItemsList = dynamic(
  () => import('@/components/fashion-items-list').then((mod) => mod.FashionItemsList),
  { ssr: false }
);

const DataUpload = dynamic(
  () => import('@/components/data-upload').then((mod) => mod.DataUpload),
  { ssr: false }
);

const ModelRetraining = dynamic(
    () => import('@/components/model-retraining'),
    { ssr: false }
  );

  const ModelMetrics = dynamic(
    () => import('@/components/model-metrics').then((mod) => mod.ModelMetrics),
    { ssr: false }
  );

export function DashboardClient() {
  return (
    <div className="space-y-6">
      <FashionPredictionForm />
      <FashionItemsList />
      <DataUpload />
      <ModelRetraining />
      <ModelMetrics />
    </div>
  );
}