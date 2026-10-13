package main

import (
	"os"
	"os/signal"
	"syscall"
)

// waitForSignal blocks until the process is asked to stop, so in-flight work
// can finish rather than being killed mid-ffmpeg.
func waitForSignal() {
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGINT, syscall.SIGTERM)
	<-c
}
