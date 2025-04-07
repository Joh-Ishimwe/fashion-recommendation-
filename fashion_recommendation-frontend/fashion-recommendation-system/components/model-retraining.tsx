"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Loader2, RefreshCw, CheckCircle2, AlertCircle } from "lucide-react"
import { Progress } from "@/components/ui/progress"

function ModelRetraining() {
  const [isRetraining, setIsRetraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<{ success: boolean; message: string } | null>(null)

  const handleRetrain = async () => {
    setIsRetraining(true)
    setProgress(0)
    setResult(null)

    // Simulate progress updates
    const progressInterval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 95) {
          clearInterval(progressInterval)
          return 95
        }
        return prev + Math.floor(Math.random() * 5) + 1
      })
    }, 500)

    try {
      const response = await fetch("https://fashion-recommendation-og63.onrender.com/retrain/", {
        method: "POST",
      })

      clearInterval(progressInterval)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Retraining failed")
      }

      setProgress(100)
      const data = await response.json()
      setResult({
        success: true,
        message: data.message,
      })
    } catch (err) {
      setProgress(0)
      setResult({
        success: false,
        message: err instanceof Error ? err.message : "An error occurred during retraining",
      })
    } finally {
      clearInterval(progressInterval)
      setIsRetraining(false)
    }
  }

  return (
    <Card className="w-full max-w-3xl mx-auto">
      <CardHeader>
        <CardTitle>Model Retraining</CardTitle>
        <CardDescription>Retrain the machine learning model using the latest data from the database</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
      <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-lg">
          <div className="text-center space-y-4">
            <RefreshCw className="h-12 w-12 mx-auto text-pink-500" />
            <h3 className="text-lg font-medium">Retrain Fashion Recommendation Model</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Retraining will use all the fashion items data stored in the database to create a new model version. This
              process may take several minutes depending on the amount of data.
            </p>
          </div>
        </div>

        {isRetraining && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Retraining Progress</span>
              <span>{progress}%</span>
            </div>
            <Progress value={progress} className="h-2" />
            <p className="text-sm text-gray-500 text-center mt-2">
              {progress < 30
                ? "Preparing data..."
                : progress < 60
                  ? "Training model..."
                  : progress < 90
                    ? "Evaluating model performance..."
                    : "Finalizing model..."}
            </p>
          </div>
        )}

        <Button onClick={handleRetrain} className="w-full" disabled={isRetraining}>
          {isRetraining ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Retraining Model...
            </>
          ) : (
            "Retrain Model"
          )}
        </Button>
      </CardContent>
      <CardFooter className="flex flex-col">
        {result && (
          <Alert
            className={`mt-4 ${
              result.success
                ? "border-green-200 bg-green-50 dark:border-green-900 dark:bg-green-950"
                : "border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950"
            }`}
            variant={result.success ? "default" : "destructive"}
          >
            {result.success ? (
              <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400" />
            ) : (
              <AlertCircle className="h-4 w-4" />
            )}
            <AlertTitle className={result.success ? "text-green-800 dark:text-green-300" : ""}>
              {result.success ? "Success" : "Error"}
            </AlertTitle>
            <AlertDescription className={result.success ? "text-green-700 dark:text-green-400" : ""}>
              {result.message}
            </AlertDescription>
          </Alert>
        )}
      </CardFooter>
    </Card>
  )
}

export default ModelRetraining;