import { feature } from 'bun:bundle'
import * as React from 'react'
import { useState } from 'react'
import { resetCostState } from '../../bootstrap/state.js'
import {
  clearTrustedDeviceToken,
  enrollTrustedDevice,
} from '../../bridge/trustedDevice.js'
import type { LocalJSXCommandContext } from '../../commands.js'
import { ConfigurableShortcutHint } from '../../components/ConfigurableShortcutHint.js'
import { Dialog } from '../../components/design-system/Dialog.js'
import TextInput from '../../components/TextInput.js'
import { useMainLoopModel } from '../../hooks/useMainLoopModel.js'
import { useTerminalSize } from '../../hooks/useTerminalSize.js'
import { Box, Text } from '../../ink.js'
import { refreshGrowthBookAfterAuthChange } from '../../services/analytics/growthbook.js'
import { verifyApiKey } from '../../services/api/claude.js'
import { refreshPolicyLimits } from '../../services/policyLimits/index.js'
import { refreshRemoteManagedSettings } from '../../services/remoteManagedSettings/index.js'
import type { LocalJSXCommandOnDone } from '../../types/command.js'
import { saveApiKey } from '../../utils/auth.js'
import { stripSignatureBlocks } from '../../utils/messages.js'
import {
  checkAndDisableAutoModeIfNeeded,
  checkAndDisableBypassPermissionsIfNeeded,
  resetAutoModeGateCheck,
  resetBypassPermissionsCheck,
} from '../../utils/permissions/bypassPermissionsKillswitch.js'
import { resetUserCache } from '../../utils/user.js'

export async function call(
  onDone: LocalJSXCommandOnDone,
  context: LocalJSXCommandContext,
): Promise<React.ReactNode> {
  return (
    <Login
      onDone={async success => {
        context.onChangeAPIKey()
        // Signature-bearing blocks are bound to the API key. Strip them so the
        // new Pincode key does not reject stale signatures.
        context.setMessages(stripSignatureBlocks)
        if (success) {
          resetCostState()
          void refreshRemoteManagedSettings()
          void refreshPolicyLimits()
          resetUserCache()
          refreshGrowthBookAfterAuthChange()
          clearTrustedDeviceToken()
          void enrollTrustedDevice()
          resetBypassPermissionsCheck()
          const appState = context.getAppState()
          void checkAndDisableBypassPermissionsIfNeeded(
            appState.toolPermissionContext,
            context.setAppState,
          )
          if (feature('TRANSCRIPT_CLASSIFIER')) {
            resetAutoModeGateCheck()
            void checkAndDisableAutoModeIfNeeded(
              appState.toolPermissionContext,
              context.setAppState,
              appState.fastMode,
            )
          }
          context.setAppState(prev => ({
            ...prev,
            authVersion: prev.authVersion + 1,
          }))
        }
        onDone(success ? 'Pincode API key saved' : 'Login interrupted')
      }}
    />
  )
}

export function PincodeApiKeyForm(props: {
  onDone: (success: boolean) => void
  startingMessage?: string
}): React.ReactNode {
  const [apiKey, setApiKey] = useState('')
  const [cursorOffset, setCursorOffset] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [isSaving, setIsSaving] = useState(false)
  const terminalSize = useTerminalSize()

  async function handleSubmit(value = apiKey) {
    const trimmed = value.trim()
    if (!trimmed) {
      setError('Enter a Pincode API key.')
      return
    }

    setIsSaving(true)
    setError(null)
    try {
      const valid = await verifyApiKey(trimmed, false)
      if (!valid) {
        setError('Pincode API key was rejected. Paste a valid local vLLM API key.')
        setIsSaving(false)
        return
      }
      await saveApiKey(trimmed)
      props.onDone(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save API key.')
      setIsSaving(false)
    }
  }

  return (
    <Box flexDirection="column" gap={1}>
      {props.startingMessage && <Text>{props.startingMessage}</Text>}
      <Text>Enter your local vLLM API key (optional).</Text>
      <Text dimColor>
        This is the API key for your local vLLM endpoint. It is stored
        locally and used for local inference requests.
      </Text>
      <TextInput
        value={apiKey}
        onChange={value => {
          setApiKey(value)
          if (error) setError(null)
        }}
        onSubmit={handleSubmit}
        onPaste={setApiKey}
        focus={true}
        placeholder="sk-..."
        mask="*"
        columns={terminalSize.columns}
        cursorOffset={cursorOffset}
        onChangeCursorOffset={setCursorOffset}
        showCursor={true}
      />
      {isSaving && <Text dimColor>Validating Pincode API key...</Text>}
      {error && <Text color="error">{error}</Text>}
    </Box>
  )
}

export function Login(props: {
  onDone: (success: boolean, mainLoopModel: string) => void
  startingMessage?: string
}): React.ReactNode {
  const mainLoopModel = useMainLoopModel()
  const handleCancel = () => props.onDone(false, mainLoopModel)
  const handleDone = (success: boolean) => props.onDone(success, mainLoopModel)

  return (
    <Dialog
      title="Pincode API key"
      onCancel={handleCancel}
      color="permission"
      inputGuide={exitState =>
        exitState.pending ? (
          <Text>Press {exitState.keyName} again to exit</Text>
        ) : (
          <ConfigurableShortcutHint
            action="confirm:no"
            context="Confirmation"
            fallback="Esc"
            description="cancel"
          />
        )
      }
    >
      <PincodeApiKeyForm
        onDone={handleDone}
        startingMessage={props.startingMessage}
      />
    </Dialog>
  )
}
