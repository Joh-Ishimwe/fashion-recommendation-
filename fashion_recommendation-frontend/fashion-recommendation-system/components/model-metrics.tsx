"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Loader2, RefreshCw } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface ModelMetrics {
  current_version: number
  classification_report: {
    [key: string]: {
      precision: number
      recall: number
      "f1-score": number
      support: number
    } | number
  }
  accuracy: number
  f1_score: number
  precision: number
  recall: number
  confusion_matrix: number[][]
  training_samples: number
  test_samples: number
  timestamp: string
}

export function ModelMetrics() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchMetrics = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch("https://fashion-recommendation-og63.onrender.com/metrics/")
      if (!response.ok) {
        throw new Error("Failed to fetch model metrics")
      }
      const data = await response.json()
      setMetrics(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchMetrics()
  }, [])

  const formatPercentage = (value: number) => {
    return (value * 100).toFixed(2) + "%"
  }

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString()
  }

  const classNames = metrics 
    ? Object.keys(metrics.classification_report).filter(
        key => !["accuracy", "macro avg", "weighted avg"].includes(key))
    : []

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Model Performance Metrics</CardTitle>
          <CardDescription>
            Version {metrics?.current_version} • Last updated: {metrics ? formatDate(metrics.timestamp) : 'N/A'}
          </CardDescription>
        </div>
        <Button variant="outline" size="sm" onClick={fetchMetrics} disabled={loading}>
          {loading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <RefreshCw className="h-4 w-4 mr-2" />}
          Refresh
        </Button>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex justify-center items-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
          </div>
        ) : error ? (
          <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : metrics ? (
          <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <MetricCard 
                title="Accuracy" 
                value={formatPercentage(metrics.accuracy)} 
                description="Overall model accuracy"
              />
              <MetricCard 
                title="Precision" 
                value={formatPercentage(metrics.precision)} 
                description="Weighted average precision"
              />
              <MetricCard 
                title="Recall" 
                value={formatPercentage(metrics.recall)} 
                description="Weighted average recall"
              />
              <MetricCard 
                title="F1 Score" 
                value={formatPercentage(metrics.f1_score)} 
                description="Weighted average F1 score"
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <MetricCard 
                title="Training Samples" 
                value={metrics.training_samples.toLocaleString()} 
                description="Number of training examples"
              />
              <MetricCard 
                title="Test Samples" 
                value={metrics.test_samples.toLocaleString()} 
                description="Number of test examples"
              />
            </div>

            <div className="border rounded-lg overflow-hidden">
              <div className="bg-gray-50 dark:bg-gray-800 px-6 py-3 border-b">
                <h3 className="font-medium">Classification Report</h3>
              </div>
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Class</TableHead>
                      <TableHead>Precision</TableHead>
                      <TableHead>Recall</TableHead>
                      <TableHead>F1 Score</TableHead>
                      <TableHead>Support</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {classNames.map((className) => {
                      const values = metrics.classification_report[className]
                      return typeof values === 'object' ? (
                        <TableRow key={className}>
                          <TableCell className="font-medium">{className}</TableCell>
                          <TableCell>{formatPercentage(values.precision)}</TableCell>
                          <TableCell>{formatPercentage(values.recall)}</TableCell>
                          <TableCell>{formatPercentage(values["f1-score"])}</TableCell>
                          <TableCell>{values.support}</TableCell>
                        </TableRow>
                      ) : null
                    })}
                  </TableBody>
                </Table>
              </div>
            </div>

            <div className="border rounded-lg overflow-hidden">
              <div className="bg-gray-50 dark:bg-gray-800 px-6 py-3 border-b">
                <h3 className="font-medium">Confusion Matrix</h3>
              </div>
              <div className="p-4 overflow-x-auto">
                <div className="inline-block">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Actual \ Predicted</TableHead>
                        {classNames.map((name) => (
                          <TableHead key={name} className="text-center">{name}</TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {metrics.confusion_matrix.map((row, rowIndex) => (
                        <TableRow key={rowIndex}>
                          <TableCell className="font-medium">{classNames[rowIndex]}</TableCell>
                          {row.map((cell, cellIndex) => (
                            <TableCell 
                              key={cellIndex} 
                              className={`text-center ${
                                rowIndex === cellIndex 
                                  ? "bg-green-50 dark:bg-green-900/30 font-medium" 
                                  : cell > 0 
                                    ? "bg-red-50 dark:bg-red-900/20" 
                                    : ""
                              }`}
                            >
                              {cell}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                <div className="mt-4 text-sm text-gray-500 dark:text-gray-400">
                  <p>Diagonal cells (green) show correct predictions. Off-diagonal cells show misclassifications.</p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">No metrics available</div>
        )}
      </CardContent>
    </Card>
  )
}

function MetricCard({ title, value, description }: { title: string; value: string; description: string }) {
  return (
    <div className="border rounded-lg p-4">
      <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</h4>
      <p className="text-2xl font-bold mt-1">{value}</p>
      <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">{description}</p>
    </div>
  )
}

const ModelMetricsComponent = ModelMetrics
export default ModelMetricsComponent