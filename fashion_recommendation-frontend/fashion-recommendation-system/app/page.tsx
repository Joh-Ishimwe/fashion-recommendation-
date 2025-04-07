import type React from "react"
import { Button } from "@/components/ui/button"
import { ArrowRight, Sparkles, Zap, BarChart3, RefreshCw } from 'lucide-react'
import Link from "next/link"

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 z-0 bg-[url('/placeholder.svg?height=1080&width=1920')] bg-cover bg-center opacity-10"></div>
        <div className="container mx-auto px-4 py-16 md:py-24 relative z-10">
          <div className="flex flex-col items-center text-center">
            <div className="inline-flex items-center justify-center p-2 bg-pink-100 rounded-full mb-6 dark:bg-pink-900">
              <Sparkles className="h-6 w-6 text-pink-500" />
            </div>
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6">Fashion Recommendation System</h1>
            <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 max-w-3xl mb-10">
              Discover the perfect style recommendations powered by machine learning
            </p>
            <Link href="/dashboard" passHref>
              <Button size="lg" className="px-8 py-6 text-lg">
                Get Started <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </Link>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="container mx-auto px-4 py-16">
        <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          <FeatureCard
            icon={<Sparkles className="h-10 w-10 text-pink-500" />}
            title="Smart Predictions"
            description="Get personalized usage recommendations for fashion items based on their attributes"
          />
          <FeatureCard
            icon={<Zap className="h-10 w-10 text-pink-500" />}
            title="Instant Results"
            description="Receive immediate style predictions to enhance your fashion decision-making"
          />
          <FeatureCard
            icon={<BarChart3 className="h-10 w-10 text-pink-500" />}
            title="Performance Metrics"
            description="View detailed model performance statistics to understand prediction accuracy"
          />
          <FeatureCard
            icon={<RefreshCw className="h-10 w-10 text-pink-500" />}
            title="Continuous Learning"
            description="Upload new data and retrain the model to improve recommendations over time"
          />
        </div>
      </div>

      {/* About Section */}
      <div className="container mx-auto px-4 py-16 bg-white dark:bg-gray-900 rounded-lg shadow-sm my-16">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-8">About This Project</h2>
          <p className="text-lg text-gray-600 dark:text-gray-300 mb-6">
            This fashion recommendation system uses machine learning to classify fashion items based on key attributes,
            providing personalized outfit suggestions tailored to user preferences and occasions.
          </p>
          <p className="text-lg text-gray-600 dark:text-gray-300 mb-6">
            Our model analyzes attributes like gender, category, color, and season to predict the most suitable usage
            category for each item, helping users make better fashion choices.
          </p>
          <div className="text-center mt-10">
            <Link href="/dashboard" passHref>
              <Button size="lg" variant="outline" className="px-8">
                Explore the Dashboard
              </Button>
            </Link>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="container mx-auto px-4 py-8 border-t border-gray-200 dark:border-gray-800 mt-auto">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center space-x-2 mb-4 md:mb-0">
            <Sparkles className="h-5 w-5 text-pink-500" />
            <span className="font-semibold">Fashion Recommendation System</span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            © {new Date().getFullYear()} All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  )
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode
  title: string
  description: string
}) {
  return (
    <div className="bg-white dark:bg-gray-900 p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow">
      <div className="mb-4">{icon}</div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-gray-600 dark:text-gray-300">{description}</p>
    </div>
  )
}