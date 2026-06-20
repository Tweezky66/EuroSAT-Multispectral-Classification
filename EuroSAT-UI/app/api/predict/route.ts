import { NextResponse } from "next/server"

// URL of the FastAPI backend (app.py). Configure PYTHON_API_URL in your env
// to point at your deployed/running FastAPI server. Defaults to local dev.
const PYTHON_API_URL = process.env.PYTHON_API_URL ?? "http://localhost:8000"

export async function POST(request: Request) {
  try {
    const incoming = await request.formData()
    const file = incoming.get("file")

    if (!(file instanceof File)) {
      return NextResponse.json({ error: "No file provided." }, { status: 400 })
    }

    if (!/\.(tif|tiff)$/i.test(file.name)) {
      return NextResponse.json(
        { error: "Only .tiff / .tif multispectral files are supported." },
        { status: 400 },
      )
    }

    const forwarded = new FormData()
    forwarded.append("file", file, file.name)

    const res = await fetch(`${PYTHON_API_URL}/predict/`, {
      method: "POST",
      body: forwarded,
    })

    const data = await res.json().catch(() => null)

    if (!res.ok) {
      return NextResponse.json(
        { error: data?.detail ?? `Backend error (${res.status}).` },
        { status: res.status },
      )
    }

    return NextResponse.json(data)
  } catch (err) {
    const message =
      err instanceof Error && err.message.includes("fetch")
        ? "Could not reach the FastAPI backend. Is it running and is PYTHON_API_URL set?"
        : err instanceof Error
          ? err.message
          : "Unexpected error."
    return NextResponse.json({ error: message }, { status: 502 })
  }
}
