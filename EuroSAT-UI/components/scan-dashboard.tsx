"use client"

import type React from "react"

import { useCallback, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { ScanPanel } from "@/components/scan-panel"
import { Leaf, Satellite, Upload, Loader2, AlertTriangle } from "lucide-react"

interface PredictResult {
  filename: string
  predicted_class_index: number
  predicted_class: string
  confidence: number
  vegetation: {
    mean_ndvi: number
    health_label: string
  }
  images: {
    rgb: string
    ndvi: string
    gradcam: string
  }
}

function ndviColor(ndvi: number) {
  if (ndvi >= 0.4) return "text-chart-3"
  if (ndvi >= 0.2) return "text-chart-2"
  if (ndvi >= 0) return "text-chart-4"
  return "text-chart-5"
}

export function ScanDashboard() {
  const [result, setResult] = useState<PredictResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const runScan = useCallback(async (file: File) => {
    if (!/\.(tif|tiff)$/i.test(file.name)) {
      setError("Please upload a .tiff / .tif multispectral file.")
      return
    }
    setError(null)
    setLoading(true)
    setFileName(file.name)
    try {
      const formData = new FormData()
      formData.append("file", file)
      const res = await fetch("/api/predict", { method: "POST", body: formData })
      if (!res.ok) {
        const detail = await res.json().catch(() => null)
        throw new Error(detail?.detail || detail?.error || `Request failed (${res.status})`)
      }
      const data = (await res.json()) as PredictResult
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong while scanning.")
      setResult(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) runScan(file)
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    const file = e.dataTransfer.files?.[0]
    if (file) runScan(file)
  }

  const confidencePct = result ? Math.round(result.confidence * 100) : 0

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-8 md:py-12">
      <header className="flex flex-col gap-2">
        <div className="flex items-center gap-2 text-primary">
          <img
            src="/earth-logo.png"
            alt="TerraScan earth logo"
            className="size-7 rounded-full ring-1 ring-primary/30"
          />
          <span className="text-xs font-semibold uppercase tracking-widest">TerraScan</span>
        </div>
        <h1 className="text-balance text-2xl font-semibold tracking-tight md:text-3xl">
          Multispectral Crop Health Monitoring
        </h1>
        <p className="max-w-2xl text-pretty text-sm leading-relaxed text-muted-foreground">
          Upload a Sentinel-2 multispectral <span className="font-medium text-foreground">.tiff</span> scene. TerraScan
          derives a true-color RGB view, an NIR-based NDVI vegetation health map, and a Grad-CAM land-cover
          classification side by side.
        </p>
      </header>

      {/* Uploader */}
      <Card>
        <CardContent className="pt-6">
          <label
            htmlFor="tiff-input"
            onDragOver={(e) => {
              e.preventDefault()
              setDragActive(true)
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
            className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors ${
              dragActive ? "border-primary bg-accent" : "border-border hover:border-primary/60 hover:bg-accent/40"
            }`}
          >
            <span className="flex size-12 items-center justify-center rounded-full bg-primary/15 text-primary">
              <Upload className="size-5" aria-hidden="true" />
            </span>
            <span className="text-sm font-medium">Drop a .tiff scene here, or click to browse</span>
            <span className="text-xs text-muted-foreground">
              Sentinel-2 bands expected: 4=Red, 3=Green, 2=Blue, 8=NIR
            </span>
            <input
              id="tiff-input"
              ref={inputRef}
              type="file"
              accept=".tif,.tiff"
              className="sr-only"
              onChange={onFileChange}
            />
            <Button
              type="button"
              variant="secondary"
              size="sm"
              className="mt-1"
              onClick={(e) => {
                e.preventDefault()
                inputRef.current?.click()
              }}
            >
              Choose file
            </Button>
          </label>

          {fileName ? (
            <p className="mt-3 truncate text-center text-xs text-muted-foreground">
              {loading ? "Scanning" : "Loaded"}: <span className="text-foreground">{fileName}</span>
            </p>
          ) : null}

          {error ? (
            <div
              role="alert"
              className="mt-4 flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive"
            >
              <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
              <span>{error}</span>
            </div>
          ) : null}
        </CardContent>
      </Card>

      {loading ? (
        <div className="flex items-center justify-center gap-2 py-10 text-sm text-muted-foreground">
          <Loader2 className="size-4 animate-spin" aria-hidden="true" />
          Analyzing multispectral bands…
        </div>
      ) : null}

      {result && !loading ? (
        <>
          {/* Prediction summary */}
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Satellite className="size-4 text-primary" aria-hidden="true" />
                  Land-cover classification
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-3">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-2xl font-semibold tracking-tight text-balance">{result.predicted_class}</span>
                  <span className="text-sm font-medium text-muted-foreground">{confidencePct}%</span>
                </div>
                <Progress value={confidencePct} aria-label="Model confidence" />
                <p className="text-xs text-muted-foreground">Model confidence for the predicted class.</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Leaf className="size-4 text-primary" aria-hidden="true" />
                  Vegetation health (NDVI)
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-3">
                <div className="flex items-baseline justify-between gap-2">
                  <span className={`text-2xl font-semibold tracking-tight ${ndviColor(result.vegetation.mean_ndvi)}`}>
                    {result.vegetation.mean_ndvi.toFixed(3)}
                  </span>
                  <span className="text-right text-sm font-medium text-muted-foreground text-pretty">
                    {result.vegetation.health_label}
                  </span>
                </div>
                <Progress
                  value={Math.round(((result.vegetation.mean_ndvi + 1) / 2) * 100)}
                  aria-label="Mean NDVI"
                />
                <p className="text-xs text-muted-foreground">
                  Mean NDVI across the scene, from -1 (bare/water) to +1 (dense vegetation).
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Side-by-side imagery */}
          <div className="grid gap-4 md:grid-cols-3">
            <ScanPanel
              title="True-color RGB"
              description="Visible bands (4-3-2) for a natural reference of what the field looks like."
              src={result.images.rgb}
              alt="True-color RGB rendering derived from the uploaded .tiff visible bands"
              badge="Bands 4·3·2"
            />
            <ScanPanel
              title="NIR vegetation health"
              description="NDVI from NIR + Red. Green = healthy canopy, red = bare soil or stress."
              src={result.images.ndvi}
              alt="NDVI vegetation health heatmap derived from NIR and Red bands"
              badge="NDVI"
            />
            <ScanPanel
              title="Grad-CAM attention"
              description="Where the classifier focused when predicting the land-cover class."
              src={result.images.gradcam}
              alt="Grad-CAM attention overlay for the predicted land-cover class"
              badge="Explainability"
            />
          </div>
        </>
      ) : null}

      {!result && !loading && !error ? (
        <div className="grid gap-4 md:grid-cols-3">
          <ScanPanel title="True-color RGB" description="Visible bands 4-3-2." alt="" badge="Bands 4·3·2" />
          <ScanPanel title="NIR vegetation health" description="NDVI from NIR + Red." alt="" badge="NDVI" />
          <ScanPanel title="Grad-CAM attention" description="Classifier focus map." alt="" badge="Explainability" />
        </div>
      ) : null}
    </div>
  )
}
