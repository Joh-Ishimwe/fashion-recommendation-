"use client"

import { useState, useEffect, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Loader2, Search, Filter } from "lucide-react"
import { Badge } from "@/components/ui/badge"

interface FashionItem {
  _id: string
  gender: string
  masterCategory: string
  subCategory: string
  articleType: string
  baseColour: string
  season: string
  year: number
  usage: string
}

export function FashionItemsList() {
  const [items, setItems] = useState<FashionItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [gender, setGender] = useState<string>("all")
  const [searchTerm, setSearchTerm] = useState<string>("")
  const [limit, setLimit] = useState<number>(10)

  const fetchItems = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      let url = `https://fashion-recommendation-og63.onrender.com/fashion_items/?limit=${limit}`
      if (gender && gender !== "all") {
        url += `&gender=${gender}`
      }
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error("Failed to fetch fashion items")
      }
      const data = await response.json()
      setItems(data.items)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }, [gender, limit])

  useEffect(() => {
    fetchItems()
  }, [fetchItems])

  const filteredItems = items.filter((item) => {
    if (!searchTerm) return true
    const searchLower = searchTerm.toLowerCase()
    return (
      item.articleType.toLowerCase().includes(searchLower) ||
      item.baseColour.toLowerCase().includes(searchLower) ||
      item.usage.toLowerCase().includes(searchLower)
    )
  })

  const getUsageBadgeColor = (usage: string) => {
    const colors: Record<string, string> = {
      Casual: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300",
      Formal: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300",
      Sports: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300",
      Ethnic: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-300",
      Party: "bg-pink-100 text-pink-800 dark:bg-pink-900 dark:text-pink-300",
      "Smart Casual": "bg-teal-100 text-teal-800 dark:bg-teal-900 dark:text-teal-300",
      Travel: "bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-300",
      Home: "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300",
    }
    return colors[usage] || "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300"
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Fashion Items</CardTitle>
        <CardDescription>Browse fashion items stored in the database</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-gray-500 dark:text-gray-400" />
            <Input
              placeholder="Search by article type, color, or usage..."
              className="pl-8"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="flex gap-4">
            <div className="w-40">
              <Select value={gender} onValueChange={setGender}>
                <SelectTrigger>
                  <div className="flex items-center">
                    <Filter className="mr-2 h-4 w-4" />
                    <SelectValue placeholder="All Genders" />
                  </div>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Genders</SelectItem>
                  <SelectItem value="Men">Men</SelectItem>
                  <SelectItem value="Women">Women</SelectItem>
                  <SelectItem value="Unisex">Unisex</SelectItem>
                  <SelectItem value="Boys">Boys</SelectItem>
                  <SelectItem value="Girls">Girls</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="w-32">
              <Select value={limit.toString()} onValueChange={(value) => setLimit(Number.parseInt(value))}>
                <SelectTrigger>
                  <SelectValue placeholder="Limit" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="10">10 items</SelectItem>
                  <SelectItem value="20">20 items</SelectItem>
                  <SelectItem value="50">50 items</SelectItem>
                  <SelectItem value="100">100 items</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button variant="outline" onClick={fetchItems}>
              Refresh
            </Button>
          </div>
        </div>

        {loading ? (
          <div className="flex justify-center items-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-gray-500" />
          </div>
        ) : error ? (
          <div className="text-center py-8 text-red-500">{error}</div>
        ) : filteredItems.length === 0 ? (
          <div className="text-center py-8 text-gray-500">No fashion items found</div>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Article Type</TableHead>
                  <TableHead>Gender</TableHead>
                  <TableHead>Category</TableHead>
                  <TableHead>Color</TableHead>
                  <TableHead>Season</TableHead>
                  <TableHead>Year</TableHead>
                  <TableHead>Usage</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredItems.map((item) => (
                  <TableRow key={item._id}>
                    <TableCell className="font-medium">{item.articleType}</TableCell>
                    <TableCell>{item.gender}</TableCell>
                    <TableCell>
                      {item.masterCategory} / {item.subCategory}
                    </TableCell>
                    <TableCell>{item.baseColour}</TableCell>
                    <TableCell>{item.season}</TableCell>
                    <TableCell>{item.year}</TableCell>
                    <TableCell>
                      <Badge className={getUsageBadgeColor(item.usage)}>{item.usage}</Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// Fixed export
export default FashionItemsList