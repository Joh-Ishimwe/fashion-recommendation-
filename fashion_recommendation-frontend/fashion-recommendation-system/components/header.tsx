import { Sparkles } from 'lucide-react';
import Link from 'next/link';

// Either keep it as a named export (option 1):
export function Header() {
  return (
    <header className="bg-white dark:bg-gray-950 border-b border-gray-200 dark:border-gray-800">
      <div className="container mx-auto py-4 px-4">
        <div className="flex items-center justify-between">
          <Link href="/" className="flex items-center space-x-2">
            <Sparkles className="h-6 w-6 text-pink-500" />
            <h1 className="text-2xl font-bold">Fashion Recommendation System</h1>
          </Link>
          <div className="text-sm text-gray-500 dark:text-gray-400">
            ML-Powered Style Predictions
          </div>
        </div>
      </div>
    </header>
  );
}
