"use client";

import React, { useState, useEffect, useCallback } from "react";
import { CheckCircle2, AlertCircle, RefreshCw, Download, SkipForward, ArrowRight, User } from "lucide-react";

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

const API_ENDPOINTS = {
  nextImage: "/next-image",
  submitLabel: "/submit-label",
  exportCsv: "/export-csv",
} as const;

type Severity = "none" | "mild" | "moderate" | "severe";

interface ProblemRow {
  key: "crowding" | "gaps" | "discoloration" | "gum_line" | "symmetry";
  label: string;
}

const PROBLEMS: ProblemRow[] = [
  { key: "crowding", label: "Tooth Alignment / Crowding" },
  { key: "gaps", label: "Gaps Between Teeth" },
  { key: "discoloration", label: "Tooth Discoloration" },
  { key: "gum_line", label: "Gum Line" },
  { key: "symmetry", label: "Smile Symmetry" },
];

const SEVERITIES: { value: Severity; label: string }[] = [
  { value: "none", label: "None" },
  { value: "mild", label: "Mild" },
  { value: "moderate", label: "Moderate" },
  { value: "severe", label: "Severe" },
];

interface ImagePayload {
  filename?: string;
  url?: string;
  done?: number;
  total?: number;
  all_done?: boolean;
  message?: string;
}

export default function LabelingApp() {
  const [labeler, setLabeler] = useState<string>("");
  const [nameInput, setNameInput] = useState<string>("");
  const [isNameModalOpen, setIsNameModalOpen] = useState<boolean>(false);

  const [loading, setLoading] = useState<boolean>(true);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [currentImage, setCurrentImage] = useState<ImagePayload | null>(null);
  const [isAllDone, setIsAllDone] = useState<boolean>(false);

  // Ratings state
  const [selections, setSelections] = useState<Record<string, Severity>>({});
  const [isSkipped, setIsSkipped] = useState<boolean>(false);

  // Initialize labeler name from sessionStorage
  useEffect(() => {
    const savedName = sessionStorage.getItem("labeler_name");
    if (savedName && savedName.trim()) {
      setLabeler(savedName.trim());
    } else {
      setIsNameModalOpen(true);
    }
  }, []);

  const fetchNextImage = useCallback(async (labelerName: string) => {
    if (!labelerName) return;
    setLoading(true);
    setErrorMessage(null);
    setSelections({});
    setIsSkipped(false);

    try {
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.nextImage}?labeler=${encodeURIComponent(labelerName)}`, {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch image (HTTP ${response.status})`);
      }

      const data: ImagePayload = await response.json();

      if (data.all_done || !data.filename) {
        setIsAllDone(true);
        setCurrentImage(null);
      } else {
        setCurrentImage(data);
        setIsAllDone(false);
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unable to reach backend service.";
      setErrorMessage(message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch image whenever labeler is set
  useEffect(() => {
    if (labeler) {
      fetchNextImage(labeler);
    }
  }, [labeler, fetchNextImage]);

  const handleSaveName = (e: React.FormEvent) => {
    e.preventDefault();
    if (!nameInput.trim()) return;
    const trimmed = nameInput.trim();
    sessionStorage.setItem("labeler_name", trimmed);
    setLabeler(trimmed);
    setIsNameModalOpen(false);
  };

  const handleSelectSeverity = (key: string, value: Severity) => {
    if (isSkipped) return; // Ignore if skipped
    setSelections((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleToggleSkip = () => {
    setIsSkipped((prev) => !prev);
  };

  const isComplete = isSkipped || PROBLEMS.every((p) => selections[p.key]);

  const handleSubmit = async () => {
    if (!isComplete || !currentImage?.filename || submitting) return;

    setSubmitting(true);
    setErrorMessage(null);

    const payload = {
      filename: currentImage.filename,
      labeler: labeler,
      crowding: isSkipped ? "none" : (selections.crowding || "none"),
      gaps: isSkipped ? "none" : (selections.gaps || "none"),
      discoloration: isSkipped ? "none" : (selections.discoloration || "none"),
      gum_line: isSkipped ? "none" : (selections.gum_line || "none"),
      symmetry: isSkipped ? "none" : (selections.symmetry || "none"),
      skipped: isSkipped,
    };

    try {
      const res = await fetch(`${API_BASE_URL}${API_ENDPOINTS.submitLabel}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (${res.status})`);
      }

      // Immediately fetch next image without caching
      await fetchNextImage(labeler);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to submit label. Please try again.";
      setErrorMessage(message);
    } finally {
      setSubmitting(false);
    }
  };

  const handleExportCsv = () => {
    window.location.href = `${API_BASE_URL}${API_ENDPOINTS.exportCsv}`;
  };

  const doneCount = currentImage?.done ?? 0;
  const totalCount = currentImage?.total ?? 0;
  const progressPercent = totalCount > 0 ? Math.min(100, Math.round((doneCount / totalCount) * 100)) : 0;

  return (
    <>
      {/* Name Input Modal */}
      {isNameModalOpen && (
        <div className="modal-overlay">
          <div className="modal-card">
            <h2 className="state-title" style={{ fontSize: "1.25rem" }}>Reviewer Identification</h2>
            <p className="state-desc" style={{ marginBottom: "1rem" }}>
              Please enter your name or reviewer ID to track submissions for this session.
            </p>
            <form onSubmit={handleSaveName}>
              <input
                type="text"
                className="modal-input"
                placeholder="e.g. Dr. Emily Smith"
                value={nameInput}
                onChange={(e) => setNameInput(e.target.value)}
                autoFocus
                required
              />
              <button
                type="submit"
                className="submit-btn"
                disabled={!nameInput.trim()}
              >
                Start Reviewing
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">S</div>
          <div>
            <span className="brand-title">SmileLabel</span>
            <span className="brand-badge" style={{ marginLeft: "8px" }}>Expert Tool</span>
          </div>
        </div>

        {labeler && (
          <div className="user-badge">
            <div className="user-avatar">
              <User size={14} />
            </div>
            <span>{labeler}</span>
            <button
              className="change-user-btn"
              onClick={() => {
                setNameInput(labeler);
                setIsNameModalOpen(true);
              }}
            >
              Change
            </button>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Loading State */}
        {loading && (
          <div className="state-card">
            <div className="spinner spinner-primary" style={{ width: 40, height: 40, margin: "0 auto 1.5rem" }} />
            <h3 className="state-title">Loading Next Photo...</h3>
            <p className="state-desc">Retrieving authorized image from secure storage.</p>
          </div>
        )}

        {/* Error / Retry State */}
        {!loading && errorMessage && (
          <div className="state-card">
            <div className="state-icon error">
              <AlertCircle size={36} />
            </div>
            <h3 className="state-title">Connection Error</h3>
            <p className="state-desc">{errorMessage}</p>
            <button className="submit-btn" style={{ maxWidth: 220, margin: "0 auto" }} onClick={() => fetchNextImage(labeler)}>
              <RefreshCw size={18} />
              Retry Connection
            </button>
          </div>
        )}

        {/* All Images Completed Screen */}
        {!loading && !errorMessage && isAllDone && (
          <div className="state-card">
            <div className="state-icon success">
              <CheckCircle2 size={36} />
            </div>
            <h3 className="state-title">All Images Labeled!</h3>
            <p className="state-desc">
              All photos in this batch have been reviewed and logged. Thank you for your review.
            </p>
            <button className="submit-btn" style={{ maxWidth: 260, margin: "0 auto" }} onClick={handleExportCsv}>
              <Download size={18} />
              Export Labels (CSV)
            </button>
          </div>
        )}

        {/* Active Review Workspace */}
        {!loading && !errorMessage && !isAllDone && currentImage && (
          <>
            {/* Progress Bar Card */}
            <div className="progress-container">
              <div className="progress-header">
                <span className="progress-title">Review Progress</span>
                <span className="progress-stats">
                  Image {doneCount + 1} of {totalCount || doneCount + 1} ({progressPercent}%)
                </span>
              </div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
              </div>
            </div>

            <div className="workspace-grid">
              {/* Image Viewport */}
              <div className="image-card">
                <div className="image-header">
                  <span>Photo Preview</span>
                  <span className="image-filename" title={currentImage.filename}>
                    {currentImage.filename}
                  </span>
                </div>
                <div className="image-wrapper">
                  {currentImage.url ? (
                    // Plain img tag for signed B2 storage URLs
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={currentImage.url}
                      alt={currentImage.filename || "Smile photo to label"}
                      className="image-element"
                      loading="eager"
                    />
                  ) : (
                    <div style={{ color: "#94a3b8", fontSize: "0.9rem" }}>No image available</div>
                  )}
                </div>
              </div>

              {/* Assessment Controls */}
              <div className="labeling-card">
                {PROBLEMS.map((prob) => {
                  const currentVal = selections[prob.key];
                  return (
                    <div key={prob.key} className="criteria-group">
                      <div className="criteria-title">
                        <span>{prob.label}</span>
                        {currentVal && (
                          <span className="criteria-selected-label">{currentVal}</span>
                        )}
                      </div>
                      <div className="button-group">
                        {SEVERITIES.map((sev) => {
                          const isSelected = currentVal === sev.value;
                          const className = `rating-btn ${
                            isSelected ? `selected-${sev.value}` : ""
                          }`;
                          return (
                            <button
                              key={sev.value}
                              type="button"
                              className={className}
                              disabled={isSkipped}
                              onClick={() => handleSelectSeverity(prob.key, sev.value)}
                            >
                              {sev.label}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}

                {/* Actions */}
                <div className="actions-footer">
                  <button
                    type="button"
                    className={`skip-btn ${isSkipped ? "skip-active" : ""}`}
                    onClick={handleToggleSkip}
                  >
                    <SkipForward size={16} />
                    {isSkipped ? "Unmark Skip (Resume Rating)" : "Skip Image (Unusable / Blurry Photo)"}
                  </button>

                  <button
                    type="button"
                    className="submit-btn"
                    disabled={!isComplete || submitting}
                    onClick={handleSubmit}
                  >
                    {submitting ? (
                      <>
                        <div className="spinner" style={{ width: 18, height: 18 }} />
                        Saving & Loading Next...
                      </>
                    ) : (
                      <>
                        <span>Submit & Next</span>
                        <ArrowRight size={18} />
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </>
  );
}
