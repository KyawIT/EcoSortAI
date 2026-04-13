import { NextResponse } from "next/server";

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export async function POST(request: Request): Promise<Response> {
  try {
    const formData = await request.formData();
    const file = formData.get("file");

    if (!(file instanceof File)) {
      return NextResponse.json(
        { error: "Kein Bild empfangen. Bitte eine Datei auswaehlen." },
        { status: 400 }
      );
    }

    const backendBaseUrl =
      process.env.ECOSORT_API_BASE_URL?.trim() || DEFAULT_API_BASE_URL;

    const backendForm = new FormData();
    backendForm.append("file", file, file.name || "snapshot.jpg");

    const response = await fetch(`${backendBaseUrl}/predict?top_k=3`, {
      method: "POST",
      body: backendForm,
      cache: "no-store",
    });

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : { error: await response.text() };

    if (!response.ok) {
      const detail =
        typeof payload?.detail === "string"
          ? payload.detail
          : typeof payload?.error === "string"
            ? payload.error
            : "Backend Fehler";
      return NextResponse.json(
        {
          error: "Vorhersage fehlgeschlagen.",
          details: detail,
        },
        { status: response.status }
      );
    }

    return NextResponse.json(payload, { status: 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unbekannter Fehler";
    return NextResponse.json(
      {
        error: "Proxy-Fehler beim Verarbeiten der Anfrage.",
        details: message,
      },
      { status: 500 }
    );
  }
}
