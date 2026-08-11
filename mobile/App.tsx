import { StatusBar } from "expo-status-bar";
import React, { useState } from "react";
import { Alert, SafeAreaView, StyleSheet } from "react-native";

import { AnalyzeScreen } from "@/screens/AnalyzeScreen";
import { HomeScreen, SelectedVideo } from "@/screens/HomeScreen";
import { ResultsScreen } from "@/screens/ResultsScreen";
import type { AnalysisResult } from "@/types";

type Screen =
  | { name: "home" }
  | { name: "analyzing"; video: SelectedVideo }
  | { name: "results"; video: SelectedVideo; result: AnalysisResult };

export default function App() {
  const [screen, setScreen] = useState<Screen>({ name: "home" });

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      {screen.name === "home" && (
        <HomeScreen onVideoSelected={(video) => setScreen({ name: "analyzing", video })} />
      )}
      {screen.name === "analyzing" && (
        <AnalyzeScreen
          video={screen.video}
          onComplete={(result) => setScreen({ name: "results", video: screen.video, result })}
          onError={(message) => {
            Alert.alert("Analysis failed", message);
            setScreen({ name: "home" });
          }}
          onCancel={() => setScreen({ name: "home" })}
        />
      )}
      {screen.name === "results" && (
        <ResultsScreen
          videoUri={screen.video.uri}
          result={screen.result}
          onReset={() => setScreen({ name: "home" })}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#0B1220",
  },
});
