import { ResizeMode, Video } from "expo-av";
import React, { useEffect, useState } from "react";
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";

import { combineNarratives } from "@/api/client";
import { ActionTimeline, LABEL_TITLES } from "@/components/ActionTimeline";
import { SummaryBreakdown } from "@/components/SummaryBreakdown";
import type { AnalysisResult } from "@/types";

export interface AnalyzedPlayer {
  label: string;
  result: AnalysisResult;
}

interface Props {
  videoUri: string;
  players: AnalyzedPlayer[];
  onAddPlayer: () => void;
  onReset: () => void;
}

export function ResultsScreen({ videoUri, players, onAddPlayer, onReset }: Props) {
  const [highlightedIndex, setHighlightedIndex] = useState(players.length - 1);
  const [combinedNarrative, setCombinedNarrative] = useState("");
  const [loadingCombined, setLoadingCombined] = useState(false);

  useEffect(() => {
    setHighlightedIndex(players.length - 1);
  }, [players.length]);

  useEffect(() => {
    if (players.length < 2) {
      setCombinedNarrative("");
      return;
    }
    let cancelled = false;
    setLoadingCombined(true);
    combineNarratives({
      players: players.map((p) => ({ label: p.label, segments: p.result.segments })),
    })
      .then((response) => {
        if (!cancelled) setCombinedNarrative(response.narrative);
      })
      .catch(() => {
        if (!cancelled) setCombinedNarrative("");
      })
      .finally(() => {
        if (!cancelled) setLoadingCombined(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [players]);

  const highlighted = players[highlightedIndex] ?? players[0];
  const result = highlighted.result;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Video
        source={{ uri: videoUri }}
        style={styles.video}
        useNativeControls
        resizeMode={ResizeMode.CONTAIN}
        isLooping
      />

      {players.length > 1 && (
        <>
          <Text style={styles.sectionTitle}>Sequence of events</Text>
          <View style={styles.narrativeBox}>
            {loadingCombined ? (
              <ActivityIndicator color="#F97316" />
            ) : (
              <Text style={styles.narrativeText}>{combinedNarrative}</Text>
            )}
          </View>

          <Text style={styles.sectionTitle}>Highlighted player</Text>
          <View style={styles.playerPicker}>
            {players.map((p, index) => (
              <Pressable
                key={`${p.label}-${index}`}
                onPress={() => setHighlightedIndex(index)}
                style={[styles.playerChip, index === highlightedIndex && styles.playerChipSelected]}
              >
                <Text
                  style={[
                    styles.playerChipText,
                    index === highlightedIndex && styles.playerChipTextSelected,
                  ]}
                >
                  {p.label}
                </Text>
              </Pressable>
            ))}
          </View>
        </>
      )}

      {result.player_selected && (
        <Text style={styles.selectedNote}>🎯 Focused on the player you selected</Text>
      )}

      {!!result.narrative && (
        <>
          <Text style={styles.sectionTitle}>
            {players.length > 1 ? `Movements of ${highlighted.label}` : "Narrative"}
          </Text>
          <View style={styles.narrativeBox}>
            <Text style={styles.narrativeText}>{result.narrative}</Text>
          </View>
        </>
      )}

      <Text style={styles.sectionTitle}>Timeline</Text>
      <ActionTimeline segments={result.segments} durationSeconds={result.duration_seconds} />

      <Text style={styles.sectionTitle}>Breakdown</Text>
      <SummaryBreakdown summary={result.summary} />

      <Text style={styles.sectionTitle}>Segments</Text>
      {result.segments.map((segment, index) => (
        <View key={`${segment.start_time}-${index}`} style={styles.segmentRow}>
          <Text style={styles.segmentLabel}>
            {LABEL_TITLES[segment.label]}
            {segment.dominant_hand ? ` (${segment.dominant_hand} hand)` : ""}
          </Text>
          <Text style={styles.segmentTime}>
            {segment.start_time.toFixed(1)}s – {segment.end_time.toFixed(1)}s (
            {Math.round(segment.confidence * 100)}% confidence)
          </Text>
        </View>
      ))}

      <Pressable style={styles.addPlayerButton} onPress={onAddPlayer}>
        <Text style={styles.addPlayerText}>+ Analyze another player in this clip</Text>
      </Pressable>

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
  selectedNote: {
    color: "#9CA3AF",
    fontSize: 12,
    marginBottom: 10,
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
  playerPicker: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  playerChip: {
    borderWidth: 1,
    borderColor: "#374151",
    borderRadius: 20,
    paddingVertical: 8,
    paddingHorizontal: 14,
  },
  playerChipSelected: {
    backgroundColor: "#F97316",
    borderColor: "#F97316",
  },
  playerChipText: {
    color: "#E5E7EB",
    fontSize: 13,
    fontWeight: "600",
  },
  playerChipTextSelected: {
    color: "#0B1220",
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
  addPlayerButton: {
    marginTop: 28,
    backgroundColor: "#1F2937",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  addPlayerText: {
    color: "#F97316",
    fontSize: 15,
    fontWeight: "600",
  },
  resetButton: {
    marginTop: 12,
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
