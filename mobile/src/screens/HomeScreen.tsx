import * as ImagePicker from "expo-image-picker";
import React, { useState } from "react";
import { Alert, Pressable, StyleSheet, Text, TextInput, View } from "react-native";

export interface SelectedVideo {
  uri: string;
  fileName: string;
  jerseyNumber?: string;
}

interface Props {
  onVideoSelected: (video: SelectedVideo) => void;
}

function fileNameFromUri(uri: string): string {
  const parts = uri.split("/");
  return parts[parts.length - 1] || "clip.mp4";
}

export function HomeScreen({ onVideoSelected }: Props) {
  const [jerseyNumber, setJerseyNumber] = useState("");

  const handlePicked = (result: ImagePicker.ImagePickerResult) => {
    if (result.canceled || result.assets.length === 0) return;
    const asset = result.assets[0];
    const trimmed = jerseyNumber.trim();
    onVideoSelected({
      uri: asset.uri,
      fileName: asset.fileName ?? fileNameFromUri(asset.uri),
      jerseyNumber: trimmed || undefined,
    });
  };

  const recordVideo = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Camera permission needed", "Enable camera access to record a video.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      videoMaxDuration: 60,
      quality: 1,
    });
    handlePicked(result);
  };

  const pickFromLibrary = async () => {
    const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Library permission needed", "Enable photo library access to choose a video.");
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      quality: 1,
    });
    handlePicked(result);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Basketball Movement Analyzer</Text>
      <Text style={styles.subtitle}>
        Record or upload a clip of a player and get a breakdown of dribbling, shooting, passing,
        and off-ball movement.
      </Text>

      <Text style={styles.label}>Jersey number (optional)</Text>
      <TextInput
        style={styles.input}
        value={jerseyNumber}
        onChangeText={(text) => setJerseyNumber(text.replace(/[^0-9]/g, "").slice(0, 2))}
        placeholder="e.g. 23"
        placeholderTextColor="#6B7280"
        keyboardType="number-pad"
        maxLength={2}
      />
      <Text style={styles.hint}>
        Focus the analysis on a specific player. If we can&apos;t confidently find this number in
        the clip, we&apos;ll fall back to tracking whoever the default heuristic picks and let you
        know.
      </Text>

      <Pressable style={styles.primaryButton} onPress={recordVideo}>
        <Text style={styles.primaryButtonText}>Record a video</Text>
      </Pressable>

      <Pressable style={styles.secondaryButton} onPress={pickFromLibrary}>
        <Text style={styles.secondaryButtonText}>Choose from library</Text>
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
  },
  title: {
    fontSize: 26,
    fontWeight: "700",
    color: "#F9FAFB",
    marginBottom: 12,
    textAlign: "center",
  },
  subtitle: {
    fontSize: 15,
    color: "#9CA3AF",
    textAlign: "center",
    marginBottom: 36,
    lineHeight: 21,
  },
  label: {
    color: "#E5E7EB",
    fontSize: 14,
    fontWeight: "600",
    marginBottom: 8,
  },
  input: {
    borderWidth: 1,
    borderColor: "#374151",
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 14,
    color: "#F9FAFB",
    fontSize: 16,
    marginBottom: 8,
  },
  hint: {
    color: "#6B7280",
    fontSize: 12,
    lineHeight: 17,
    marginBottom: 24,
  },
  primaryButton: {
    backgroundColor: "#F97316",
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
    marginBottom: 14,
  },
  primaryButtonText: {
    color: "#0B1220",
    fontSize: 16,
    fontWeight: "700",
  },
  secondaryButton: {
    borderWidth: 1,
    borderColor: "#374151",
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
  },
  secondaryButtonText: {
    color: "#E5E7EB",
    fontSize: 16,
    fontWeight: "600",
  },
});
