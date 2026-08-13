import { StatusBar } from "expo-status-bar";
import React, { useState } from "react";
import { Alert, SafeAreaView, StyleSheet } from "react-native";

import { AnalyzeScreen } from "@/screens/AnalyzeScreen";
import { HomeScreen, SelectedVideo } from "@/screens/HomeScreen";
import { ResultsScreen } from "@/screens/ResultsScreen";
import { PlayerSelection, SelectPlayerScreen } from "@/screens/SelectPlayerScreen";
import type { AnalysisResult } from "@/types";

type Screen =
  | { name: "home" }
  | { name: "selectPlayer"; video: SelectedVideo }
  | { name: "analyzing"; video: SelectedVideo; selection: PlayerSelection }
  | { name: "results"; video: SelectedVideo; result: AnalysisResult };

export default function App() {
  const [screen, setScreen] = useState<Screen>({ name: "home" });

  const handleError = (message: string) => {
    Alert.alert("Something went wrong", message);
    setScreen({ name: "home" });
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      {screen.name === "home" && (
        <HomeScreen onVideoSelected={(video) => setScreen({ name: "selectPlayer", video })} />
      )}
      {screen.name === "selectPlayer" && (
        <SelectPlayerScreen
          video={screen.video}
          onContinue={(selection) =>
            setScreen({ name: "analyzing", video: screen.video, selection })
          }
          onError={handleError}
        />
      )}
      {screen.name === "analyzing" && (
        <AnalyzeScreen
          selection={screen.selection}
          onComplete={(result) => setScreen({ name: "results", video: screen.video, result })}
          onError={handleError}
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
