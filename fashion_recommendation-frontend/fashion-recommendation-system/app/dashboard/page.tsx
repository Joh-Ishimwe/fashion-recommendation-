import { Header } from "@/components/header";
import ClientOnly from "@/components/client-only";
import DashboardContent from "../dashboard-content";

export default function Dashboard() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <Header />
      <div className="container mx-auto py-8 px-4">
        <ClientOnly>
          <DashboardContent />
        </ClientOnly>
      </div>
    </main>
  );
}