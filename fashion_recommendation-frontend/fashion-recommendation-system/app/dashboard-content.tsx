'use client';

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import dynamic from "next/dynamic";


// Dynamic imports with proper loading states
const FashionPredictionForm = dynamic(
  () => import("@/components/fashion-prediction-form"),
  { 
    ssr: false,
    loading: () => <div className="p-4 text-center">Loading prediction form...</div>
  }
);

const FashionItemsList = dynamic(
  () => import("@/components/fashion-items-list").then(mod => mod.FashionItemsList),
  { 
    ssr: false,
    loading: () => <div className="p-4 text-center">Loading items list...</div>
  }
);

const DataUpload = dynamic(
  () => import("@/components/data-upload").then(mod => mod.DataUpload),
  { 
    ssr: false,
    loading: () => <div className="p-4 text-center">Loading upload form...</div>
  }
);

const ModelRetraining = dynamic(
  () => import("@/components/model-retraining"),
  { 
    ssr: false,
    loading: () => <div className="p-4 text-center">Loading retraining panel...</div>
  }
);

const ModelMetrics = dynamic(
  () => import("@/components/model-metrics").then(mod => mod.ModelMetrics),
  { 
    ssr: false,
    loading: () => <div className="p-4 text-center">Loading metrics...</div>
  }
);

export default function DashboardContent() {
  return (
    <Tabs defaultValue="predict" className="w-full">
      <TabsList className="grid w-full grid-cols-5">
        <TabsTrigger value="predict">Predict</TabsTrigger>
        <TabsTrigger value="items">Fashion Items</TabsTrigger>
        <TabsTrigger value="upload">Upload Data</TabsTrigger>
        <TabsTrigger value="retrain">Retrain Model</TabsTrigger>
        <TabsTrigger value="metrics">Model Metrics</TabsTrigger>
      </TabsList>
      
      <TabsContent value="predict" className="mt-6">
        <FashionPredictionForm />
      </TabsContent>
      
      <TabsContent value="items" className="mt-6">
        <FashionItemsList />
      </TabsContent>
      
      <TabsContent value="upload" className="mt-6">
        <DataUpload />
      </TabsContent>
      
      <TabsContent value="retrain" className="mt-6">
        <ModelRetraining />
      </TabsContent>
      
      <TabsContent value="metrics" className="mt-6">
        <ModelMetrics />
      </TabsContent>
    </Tabs>
  );
}

