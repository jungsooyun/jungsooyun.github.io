---
title: Macmini as Server
search: true
categories:
 - general
tags:
 - macOS
 - equipment
last_modified_at: 2023-08-22 20:05
classes : wide
---

With the increased support for macOS by financial and government institutions, I boldly use Macmini as a home server, thinking that macOS is sufficient. (~~There's also the reason that the Ubuntu Mini PC died~~).

Setting up a public IP is burdensome and requires a lot of configuration, so I use Tailscale as a VPN, which was free and easiest to set up, allowing ssh access only between registered devices via private IP. For development-related work, I use ssh, and for cases requiring GUI, I use the built-in Screen Sharing on macOS. 

However, there were times when the Mac mini went into sleep mode and the server did not respond to pings due to disk sleep.  I was sad to leave my Mac mini at home and come to work, but thanks to the MacOS option

```
Wake for network access:
While sleeping, your Mac can receive incoming network traffic, such as iMessages and other iCloud updates, to keep your applications up to date
```

there was an amazing escape route where I could attach via ssh when the disk was briefly activated at 3-5 minute intervals. And I found the appropriate setting to use Macmini as a server.

The script is below. (The source is included in the script.)


```
### Confiure a Mac OS X machine for live performance use
### Toby Harris - http://tobyz.net / http://sparklive.net

## Live

# stop all those damn noises
defaults write NSGlobalDomain com.apple.sound.beep.feedback -bool false

# Disable crash reporter dialog
defaults write com.apple.CrashReporter DialogType none

# Disable dashboard
defaults write com.apple.dashboard mcx-disabled -boolean YES

# Disable screensaver
defaults write com.apple.screensaver idleTime -int 0
defaults -currentHost write com.apple.screensaver idleTime -int 0

# Disable system sleep
sudo pmset sleep 0

# Disable display sleep
sudo pmset displaysleep 0

# Disable disk sleep
sudo pmset disksleep 0

# Disable Time Machine dialog when external drives mount
defaults write com.apple.TimeMachine DoNotOfferNewDisksForBackup -bool YES

## Pro-user

# Show the ~/Library folder
chflags nohidden ~/Library

# Enable wake on ethernet
sudo pmset womp 1

# Enable AirDrop over Ethernet
defaults write com.apple.NetworkBrowser BrowseAllInterfaces 1

# Reveal IP address, hostname, OS version, etc. via clock in login window
sudo defaults write /Library/Preferences/com.apple.loginwindow AdminHostInfo HostName

# Set Help Viewer windows to non-floating mode
defaults write com.apple.helpviewer DevMode -bool true

# default to graphite icon set instead of blue
defaults write NSGlobalDomain AppleAquaColorVariant -int 6

# Disable opening and closing window animations
defaults write NSGlobalDomain NSAutomaticWindowAnimationsEnabled -bool false

# Expand save panel by default
defaults write NSGlobalDomain NSNavPanelExpandedStateForSaveMode -bool true

# Expand print panel by default
defaults write NSGlobalDomain PMPrintingExpandedStateForPrint -bool true

# Save to disk (not to iCloud) by default
defaults write NSGlobalDomain NSDocumentSaveNewDocumentsToCloud -bool false

# Desktop: show external drive icons
defaults write com.apple.finder ShowExternalHardDrivesOnDesktop -bool true

# Desktop: show hard drive icons
defaults write com.apple.finder ShowHardDrivesOnDesktop -bool true

# Desktop: show mounted server icons
defaults write com.apple.finder ShowMountedServersOnDesktop -bool true

# Desktop: show removable media icons
defaults write com.apple.finder ShowRemovableMediaOnDesktop -bool true

# When performing a search, search the current folder by default
defaults write com.apple.finder FXDefaultSearchScope -string "SCcf"

# Avoid creating .DS_Store files on network volumes
defaults write com.apple.desktopservices DSDontWriteNetworkStores -bool true

# Set the icon size of Dock items to 32 pixels
defaults write com.apple.dock tilesize -int 32

# Automatically hide and show the Dock
defaults write com.apple.dock autohide -bool true

# Shorten the auto-hiding Dock delay
defaults write com.apple.dock autohide-delay -float 0.05

# Shorten the animation when hiding/showing the Dock
defaults write com.apple.dock autohide-time-modifier -float 0.25

# Enable the 2D Dock
defaults write com.apple.dock no-glass -bool true

# Show indicator lights for open applications in the Dock
defaults write com.apple.dock show-process-indicators -bool true

# Don’t animate opening applications from the Dock
defaults write com.apple.dock launchanim -bool false

# Make Dock icons of hidden applications translucent
defaults write com.apple.dock showhidden -bool true

# Use plain text mode for new documents
defaults write com.apple.TextEdit RichText -int 0

# Open files as UTF-8
defaults write com.apple.TextEdit PlainTextEncoding -int 4

# Save files as UTF-8
defaults write com.apple.TextEdit PlainTextEncodingForWrite -int 4

# Enable the debug menu in Disk Utility
defaults write com.apple.DiskUtility DUDebugMenuEnabled -bool true
defaults write com.apple.DiskUtility advanced-image-options -bool true
```


Of course, I didn't activate all the options! Use it to your taste.