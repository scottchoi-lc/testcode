import { ResizeMode, Video } from "expo-av";
import React from "react";
import { Pressable, ScrollView, StyleSheet, Text, View } from "react-native";

import { ActionTimeline, LABEL_TITLES } from "@/components/ActionTimeline";
import { SummaryBreakdown } from "@/components/SummaryBreakdown";
import type { AnalysisResult } from "@/types";

interface Props {
  videoUri: string;
  result: AnalysisResult;
  onReset: () => void;
}

export function ResultsScreen({ videoUri, result, onReset }: Props) {
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Video
        source={{ uri: videoUri }}
        style={styles.video}
        useNativeControls
        resizeMode={ResizeMode.CONTAIN}
        isLooping
      />

      {!!result.narrative && (
        <View style={styles.narrativeBox}>
          <Text style={styles.narrativeText}>{result.narrative}</Text>
        </View>
      )}

      <Text style={styles.sectionTitle}>Timeline</Text>
      <ActionTimeline segments={result.segments} durationSeconds={result.duration_seconds} />

      <Text style={styles.sectionTitle}>Breakdown</Text>
      <SummaryBreakdown summary={result.summary} />

      <Text style={styles.sectionTitle}>Segments</Text>
      {result.segments.map((segment, index) => (
        <View key={`${segment.start_time}-${index}`} style={styles.segmentRow}>
          <Text style={styles.segmentLabel}>{LABEL_TITLES[segment.label]}</Text>
          <Text style={styles.segmentTime}>
            {segment.start_time.toFixed(1)}s – {segment.end_time.toFixed(1)}s (
            {Math.round(segment.confidence * 100)}% confidence)
          </Text>
        </View>
      ))}

      <Pressable style={styles.resetButton} onPress={onReset}>
        <Text style={styles.resetText}>Analyze another clip</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0B1220",
  },
  content: {
    padding: 20,
    paddingBottom: 48,
  },
  video: {
    width: "100%",
    aspectRatio: 16 / 9,
    borderRadius: 12,
    backgroundColor: "#000",
    marginBottom: 20,
  },
  narrativeBox: {
    backgroundColor: "#111827",
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: "#1F2937",
  },
  narrativeText: {
    color: "#E5E7EB",
    fontSize: 14,
    lineHeight: 20,
  },
  sectionTitle: {
    color: "#F9FAFB",
    fontSize: 16,
    fontWeight: "700",
    marginTop: 20,
    marginBottom: 10,
  },
  segmentRow: {
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#1F2937",
    paddingVertical: 8,
  },
  segmentLabel: {
    color: "#E5E7EB",
    fontSize: 14,
    fontWeight: "600",
  },
  segmentTime: {
    color: "#9CA3AF",
    fontSize: 12,
    marginTop: 2,
  },
  resetButton: {
    marginTop: 28,
    borderWidth: 1,
    borderColor: "#374151",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  resetText: {
    color: "#E5E7EB",
    fontSize: 15,
    fontWeight: "600",
  },
});
