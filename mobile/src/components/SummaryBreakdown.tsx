import React from "react";
import { StyleSheet, Text, View } from "react-native";

import { LABEL_COLORS, LABEL_TITLES } from "@/components/ActionTimeline";
import type { ActionLabel } from "@/types";

interface Props {
  summary: Record<ActionLabel, number>;
}

export function SummaryBreakdown({ summary }: Props) {
  const entries = (Object.entries(summary) as [ActionLabel, number][])
    .filter(([, seconds]) => seconds > 0)
    .sort((a, b) => b[1] - a[1]);
  const maxSeconds = Math.max(...entries.map(([, seconds]) => seconds), 1);

  if (entries.length === 0) {
    return <Text style={styles.empty}>No actions were confidently detected in this clip.</Text>;
  }

  return (
    <View>
      {entries.map(([label, seconds]) => (
        <View key={label} style={styles.row}>
          <Text style={styles.label}>{LABEL_TITLES[label]}</Text>
          <View style={styles.barTrack}>
            <View
              style={[
                styles.barFill,
                { width: `${(seconds / maxSeconds) * 100}%`, backgroundColor: LABEL_COLORS[label] },
              ]}
            />
          </View>
          <Text style={styles.seconds}>{seconds.toFixed(1)}s</Text>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 10,
  },
  label: {
    width: 130,
    color: "#E5E7EB",
    fontSize: 13,
  },
  barTrack: {
    flex: 1,
    height: 10,
    borderRadius: 5,
    backgroundColor: "#1F2937",
    overflow: "hidden",
    marginHorizontal: 8,
  },
  barFill: {
    height: "100%",
    borderRadius: 5,
  },
  seconds: {
    width: 48,
    textAlign: "right",
    color: "#9CA3AF",
    fontSize: 12,
  },
  empty: {
    color: "#9CA3AF",
    fontSize: 13,
  },
});
