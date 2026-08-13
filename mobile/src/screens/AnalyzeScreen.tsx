import React, { useEffect, useRef, useState } from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";

import { pollJobUntilDone, submitVideoForAnalysis } from "@/api/client";
import type { PlayerSelection } from "@/screens/SelectPlayerScreen";
import type { AnalysisResult } from "@/types";

interface Props {
  selection: PlayerSelection;
  onComplete: (result: AnalysisResult) => void;
  onError: (message: string) => void;
  onCancel: () => void;
}

const STATUS_LABELS: Record<string, string> = {
  queued: "Queued...",
  processing: "Analyzing movement...",
  done: "Done!",
  failed: "Analysis failed",
};

export function AnalyzeScreen({ selection, onComplete, onError, onCancel }: Props) {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("queued");
  const cancelled = useRef({ cancelled: false });

  useEffect(() => {
    cancelled.current.cancelled = false;

    (async () => {
      try {
        const job = await submitVideoForAnalysis(
          selection.videoId,
          selection.selectedBox,
          selection.selectedTimestamp
        );
        setStatus(job.status);

        const finalJob = await pollJobUntilDone(
          job.job_id,
          (update) => {
            setProgress(update.progress);
            setStatus(update.status);
          },
          cancelled.current
        );

        if (cancelled.current.cancelled) return;

        if (finalJob.status === "done" && finalJob.result) {
          onComplete(finalJob.result);
        } else {
          onError(finalJob.error ?? "Analysis failed for an unknown reason.");
        }
      } catch (err) {
        if (!cancelled.current.cancelled) {
          const message = err instanceof Error ? err.message : "Upload failed.";
          onError(message);
        }
      }
    })();

    return () => {
      cancelled.current.cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection.videoId]);

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color="#F97316" />
      <Text style={styles.status}>{STATUS_LABELS[status] ?? status}</Text>
      <View style={styles.progressTrack}>
        <View style={[styles.progressFill, { width: `${Math.round(progress * 100)}%` }]} />
      </View>
      <Text style={styles.progressText}>{Math.round(progress * 100)}%</Text>

      <Pressable style={styles.cancelButton} onPress={onCancel}>
        <Text style={styles.cancelText}>Cancel</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0B1220",
    padding: 24,
    justifyContent: "center",
    alignItems: "center",
  },
  status: {
    color: "#F9FAFB",
    fontSize: 17,
    fontWeight: "600",
    marginTop: 20,
    marginBottom: 24,
  },
  progressTrack: {
    width: "100%",
    height: 10,
    borderRadius: 5,
    backgroundColor: "#1F2937",
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    backgroundColor: "#F97316",
    borderRadius: 5,
  },
  progressText: {
    color: "#9CA3AF",
    marginTop: 8,
    fontSize: 13,
  },
  cancelButton: {
    marginTop: 40,
    paddingVertical: 12,
    paddingHorizontal: 24,
  },
  cancelText: {
    color: "#9CA3AF",
    fontSize: 14,
  },
});
