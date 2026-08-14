import { StatusBar } from "expo-status-bar";
import React, { useState } from "react";
import { Alert, SafeAreaView, StyleSheet } from "react-native";

import { AnalyzeScreen } from "@/screens/AnalyzeScreen";
import { HomeScreen, SelectedVideo } from "@/screens/HomeScreen";
import { AnalyzedPlayer, ResultsScreen } from "@/screens/ResultsScreen";
import { PlayerSelection, SelectPlayerScreen } from "@/screens/SelectPlayerScreen";

type Screen =
  | { name: "home" }
  | { name: "selectPlayer"; video: SelectedVideo; players: AnalyzedPlayer[] }
  | { name: "analyzing"; video: SelectedVideo; selection: PlayerSelection; players: AnalyzedPlayer[] }
  | { name: "results"; video: SelectedVideo; players: AnalyzedPlayer[] };

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
        <HomeScreen
          onVideoSelected={(video) => setScreen({ name: "selectPlayer", video, players: [] })}
        />
      )}
      {screen.name === "selectPlayer" && (
        <SelectPlayerScreen
          video={screen.video}
          onContinue={(selection) =>
            setScreen({ name: "analyzing", video: screen.video, selection, players: screen.players })
          }
          onError={handleError}
        />
      )}
      {screen.name === "analyzing" && (
        <AnalyzeScreen
          selection={screen.selection}
          onComplete={(result) => {
            const label = result.detected_player_number
              ? `Number ${result.detected_player_number}`
              : `Player ${screen.players.length + 1}`;
            setScreen({
              name: "results",
              video: screen.video,
              players: [...screen.players, { label, result }],
            });
          }}
          onError={handleError}
          onCancel={() => setScreen({ name: "home" })}
        />
      )}
      {screen.name === "results" && (
        <ResultsScreen
          videoUri={screen.video.uri}
          players={screen.players}
          onAddPlayer={() =>
            setScreen({ name: "selectPlayer", video: screen.video, players: screen.players })
          }
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
