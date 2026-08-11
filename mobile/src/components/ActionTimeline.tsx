import React from "react";
import { StyleSheet, Text, View } from "react-native";

import type { ActionLabel, ActionSegment } from "@/types";

export const LABEL_COLORS: Record<ActionLabel, string> = {
  dribbling: "#3B82F6", // blue
  shooting: "#F97316", // orange
  passing: "#22C55E", // green
  moving_without_ball: "#A855F7", // purple
  idle: "#6B7280", // gray
};

export const LABEL_TITLES: Record<ActionLabel, string> = {
  dribbling: "Dribbling",
  shooting: "Shooting",
  passing: "Passing",
  moving_without_ball: "Moving (no ball)",
  idle: "No clear action",
};

interface Props {
  segments: ActionSegment[];
  durationSeconds: number;
}

export function ActionTimeline({ segments, durationSeconds }: Props) {
  const total = durationSeconds || segments[segments.length - 1]?.end_time || 1;

  return (
    <View>
      <View style={styles.bar}>
        {segments.map((segment, index) => {
          const widthPct = ((segment.end_time - segment.start_time) / total) * 100;
          return (
            <View
              key={`${segment.start_time}-${index}`}
              style={[
                styles.segment,
                { width: `${Math.max(widthPct, 0.5)}%`, backgroundColor: LABEL_COLORS[segment.label] },
              ]}
            />
          );
        })}
      </View>
      <View style={styles.legendRow}>
        {(Object.keys(LABEL_TITLES) as ActionLabel[]).map((label) => (
          <View key={label} style={styles.legendItem}>
            <View style={[styles.legendSwatch, { backgroundColor: LABEL_COLORS[label] }]} />
            <Text style={styles.legendText}>{LABEL_TITLES[label]}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  bar: {
    flexDirection: "row",
    height: 28,
    borderRadius: 6,
    overflow: "hidden",
    backgroundColor: "#1F2937",
  },
  segment: {
    height: "100%",
  },
  legendRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    marginTop: 12,
    gap: 12,
  },
  legendItem: {
    flexDirection: "row",
    alignItems: "center",
    marginRight: 12,
    marginBottom: 6,
  },
  legendSwatch: {
    width: 12,
    height: 12,
    borderRadius: 3,
    marginRight: 6,
  },
  legendText: {
    color: "#E5E7EB",
    fontSize: 13,
  },
});
