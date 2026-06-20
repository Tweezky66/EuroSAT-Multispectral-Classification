import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface ScanPanelProps {
  title: string
  description: string
  src?: string
  alt: string
  badge?: string
}

export function ScanPanel({ title, description, src, alt, badge }: ScanPanelProps) {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="gap-1 pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm font-semibold tracking-tight">{title}</CardTitle>
          {badge ? (
            <span className="rounded-full bg-secondary px-2 py-0.5 text-[11px] font-medium text-secondary-foreground">
              {badge}
            </span>
          ) : null}
        </div>
        <p className="text-xs leading-relaxed text-muted-foreground text-pretty">{description}</p>
      </CardHeader>
      <CardContent>
        <div className="aspect-square w-full overflow-hidden rounded-md border border-border bg-muted">
          {src ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={src || "/placeholder.svg"}
              alt={alt}
              className="h-full w-full object-cover"
              style={{ imageRendering: "auto" }}
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center px-4 text-center text-xs text-muted-foreground">
              Awaiting scan
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
