import { AVPlaybackStatus, ResizeMode, Video } from "expo-av";
import React, { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
  useWindowDimensions,
} from "react-native";

import { getPreviewFrame, uploadVideo } from "@/api/client";
import type { SelectedVideo } from "@/screens/HomeScreen";
import type { DetectedPersonBox, PreviewFrameResponse } from "@/types";

export interface PlayerSelection {
  videoId: string;
  selectedBox?: DetectedPersonBox;
  selectedTimestamp?: number;
}

interface Props {
  video: SelectedVideo;
  onContinue: (selection: PlayerSelection) => void;
  onError: (message: string) => void;
}

export function SelectPlayerScreen({ video, onContinue, onError }: Props) {
  const { width: screenWidth } = useWindowDimensions();
  const [videoId, setVideoId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(true);
  const [currentPosition, setCurrentPosition] = useState(0);
  const [preview, setPreview] = useState<PreviewFrameResponse | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const cancelled = useRef(false);

  useEffect(() => {
    cancelled.current = false;
    (async () => {
      try {
        const result = await uploadVideo(video.uri, video.fileName);
        if (!cancelled.current) {
          setVideoId(result.video_id);
          setUploading(false);
        }
      } catch (err) {
        if (!cancelled.current) {
          onError(err instanceof Error ? err.message : "Upload failed.");
        }
      }
    })();
    return () => {
      cancelled.current = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [video.uri]);

  const handleStatusUpdate = (status: AVPlaybackStatus) => {
    if (status.isLoaded) {
      setCurrentPosition(status.positionMillis / 1000);
    }
  };

  const useThisFrame = async () => {
    if (!videoId) return;
    setLoadingPreview(true);
    setPreview(null);
    setSelectedIndex(null);
    try {
      const result = await getPreviewFrame(videoId, currentPosition);
      setPreview(result);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Could not load that frame.");
    } finally {
      setLoadingPreview(false);
    }
  };

  const tapBox = (index: number) => {
    setSelectedIndex((current) => (current === index ? null : index));
  };

  const confirmSelection = () => {
    if (!videoId || !preview || selectedIndex === null) return;
    onContinue({
      videoId,
      selectedBox: preview.people[selectedIndex],
      selectedTimestamp: preview.timestamp,
    });
  };

  const skip = () => {
    if (!videoId) return;
    onContinue({ videoId });
  };

  const displayWidth = screenWidth - 40;
  const displayHeight = preview ? displayWidth * (preview.frame_height / preview.frame_width) : 0;
  const scale = preview ? displayWidth / preview.frame_width : 1;

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Focus on a player?</Text>
      <Text style={styles.subtitle}>
        Scrub the video to a moment where the player you want is clearly visible, tap
        &quot;Use this frame&quot;, then tap them to select and confirm. Or skip to analyze
        whoever the default tracker picks.
      </Text>

      {uploading ? (
        <View style={styles.centered}>
          <ActivityIndicator color="#F97316" />
          <Text style={styles.uploadingText}>Uploading video...</Text>
        </View>
      ) : (
        <>
          <Video
            source={{ uri: video.uri }}
            style={[styles.video, { width: displayWidth }]}
            useNativeControls
            resizeMode={ResizeMode.CONTAIN}
            onPlaybackStatusUpdate={handleStatusUpdate}
          />

          <Pressable style={styles.frameButton} onPress={useThisFrame} disabled={loadingPreview}>
            {loadingPreview ? (
              <ActivityIndicator color="#0B1220" />
            ) : (
              <Text style={styles.frameButtonText}>
                Use this frame ({currentPosition.toFixed(1)}s)
              </Text>
            )}
          </Pressable>

          {preview && (
            <View style={{ width: displayWidth, height: displayHeight, marginTop: 16 }}>
              <Image
                source={{ uri: `data:image/jpeg;base64,${preview.image_base64}` }}
                style={{ width: displayWidth, height: displayHeight, borderRadius: 12 }}
              />
              {preview.people.map((box, index) => {
                const isSelected = index === selectedIndex;
                return (
                  <Pressable
                    key={index}
                    onPress={() => tapBox(index)}
                    style={[
                      styles.playerBox,
                      isSelected && styles.playerBoxSelected,
                      {
                        left: box.x1 * scale,
                        top: box.y1 * scale,
                        width: (box.x2 - box.x1) * scale,
                        height: (box.y2 - box.y1) * scale,
                      },
                    ]}
                  >
                    {isSelected && (
                      <View style={styles.selectedBadge}>
                        <Text style={styles.selectedBadgeText}>✓ Selected</Text>
                      </View>
                    )}
                  </Pressable>
                );
              })}
              {preview.people.length === 0 && (
                <Text style={styles.noPeopleText}>
                  No players detected in this frame — try a different moment.
                </Text>
              )}
            </View>
          )}

          {preview && preview.people.length > 0 && (
            <Text style={styles.selectionHint}>
              {selectedIndex === null
                ? "Tap a player above to select them."
                : "Tap again to deselect, or continue below."}
            </Text>
          )}

          {selectedIndex !== null && (
            <Pressable style={styles.confirmButton} onPress={confirmSelection}>
              <Text style={styles.confirmButtonText}>Continue with this player</Text>
            </Pressable>
          )}

          <Pressable style={styles.skipButton} onPress={skip}>
            <Text style={styles.skipButtonText}>Skip — use default tracking</Text>
          </Pressable>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0B1220",
    padding: 20,
  },
  title: {
    fontSize: 22,
    fontWeight: "700",
    color: "#F9FAFB",
    marginBottom: 8,
    textAlign: "center",
  },
  subtitle: {
    fontSize: 13,
    color: "#9CA3AF",
    textAlign: "center",
    marginBottom: 20,
    lineHeight: 18,
  },
  centered: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  uploadingText: {
    color: "#9CA3AF",
    marginTop: 12,
    fontSize: 14,
  },
  video: {
    aspectRatio: 16 / 9,
    borderRadius: 12,
    backgroundColor: "#000",
    alignSelf: "center",
  },
  frameButton: {
    backgroundColor: "#F97316",
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
    marginTop: 14,
  },
  frameButtonText: {
    color: "#0B1220",
    fontSize: 15,
    fontWeight: "700",
  },
  playerBox: {
    position: "absolute",
    borderWidth: 3,
    borderColor: "#F97316",
    borderRadius: 6,
    backgroundColor: "rgba(249, 115, 22, 0.15)",
  },
  playerBoxSelected: {
    borderColor: "#22C55E",
    borderWidth: 4,
    backgroundColor: "rgba(34, 197, 94, 0.25)",
  },
  selectedBadge: {
    position: "absolute",
    top: -26,
    left: -4,
    backgroundColor: "#22C55E",
    borderRadius: 6,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  selectedBadgeText: {
    color: "#0B1220",
    fontSize: 11,
    fontWeight: "700",
  },
  noPeopleText: {
    color: "#9CA3AF",
    fontSize: 13,
    textAlign: "center",
    marginTop: 12,
  },
  selectionHint: {
    color: "#9CA3AF",
    fontSize: 12,
    textAlign: "center",
    marginTop: 10,
  },
  confirmButton: {
    backgroundColor: "#22C55E",
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: "center",
    marginTop: 14,
  },
  confirmButtonText: {
    color: "#0B1220",
    fontSize: 15,
    fontWeight: "700",
  },
  skipButton: {
    marginTop: 20,
    paddingVertical: 14,
    alignItems: "center",
  },
  skipButtonText: {
    color: "#9CA3AF",
    fontSize: 14,
    fontWeight: "600",
  },
});
