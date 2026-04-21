# Pincode

Terminal-native coding agent powered by Qwen 3.6 on local vLLM.

Pincode brings an interactive AI engineering workflow into your shell: chat with your codebase, edit files, run commands with approval controls, use slash commands, connect tools, and work through multi-step coding tasks from one local CLI.

## Highlights

- Qwen 3.6 as the main model
- Local vLLM OpenAI-compatible API integration
- Interactive REPL for coding, debugging, refactors, and repo exploration
- Tool use for file edits, shell commands, search, MCP servers, and project workflows
- `/login` flow for saving a local vLLM API key locally
- Clear Pincode auth status, logout, and invalid-key handling
- Pincode-themed terminal UI and status colors
- Bun-based local build with development and compiled binaries

## Requirements

- Bun 1.3.11 or newer
- macOS or Linux
- A local vLLM endpoint running Qwen 3.6

## Install

```bash
bun install
bun run build:dev
```

The development binary is created at:

```bash
./cli-dev
```

For a standard local binary:

```bash
bun run build
./cli
```

## Run Globally

To run Pincode from any terminal directory as `pincode`, build once and link the package globally:

```bash
bun install
bun run build
bun link
```

Then run it from anywhere:

```bash
pincode
```

One-shot mode also works globally:

```bash
pincode -p "summarize this repository"
```

If you rebuild later, the linked `pincode` command will use the updated `./cli` binary.

## Authenticate

Start the interactive CLI and run:

```text
/login
```

Or use environment variables:

```bash
export LOCAL_API_KEY='your-api-key'
./cli-dev
```

Check auth status:

```bash
./cli-dev auth status --json
```

Clear saved auth:

```bash
./cli-dev auth logout
```

## Usage

Start the interactive coding session:

```bash
./cli-dev
```

Ask a one-shot question:

```bash
./cli-dev -p "summarize this repository"
```

Run from source:

```bash
bun run dev
```

## Common Commands

```bash
# Install dependencies
bun install

# Standard build: ./cli
bun run build

# Development build: ./cli-dev
bun run build:dev

# Development build with experimental feature set
bun run build:dev:full

# Compiled build: ./dist/cli
bun run compile

# Run from source
bun run dev
```

## Configuration

Fabric uses a local vLLM OpenAI-compatible endpoint:

```text
http://127.0.0.1:8000/v1
```

Primary environment variables:

| Variable | Purpose |
|---|---|
| `LOCAL_API_KEY` | Local vLLM API key |
| `LOCAL_BASE_URL` | Custom compatible endpoint (default: `http://127.0.0.1:8000/v1`) |
| `ANTHROPIC_BASE_URL` | Optional custom compatible endpoint |
| `CLAUDE_CODE_MAX_RETRIES` | Override API retry count |
| `API_TIMEOUT_MS` | Override request timeout |

## Project Structure

```text
src/
  entrypoints/       CLI bootstrap
  screens/           Ink/React terminal screens
  components/        Reusable terminal UI
  commands/          Slash command implementations
  services/          API clients, auth, MCP, analytics stubs
  tools/             Tool registry and tool implementations
  utils/             Shared model, auth, config, and formatting utilities
  state/             App state store
  hooks/             React hooks used by the terminal UI
```

## Development Notes

- The main interactive UI lives in `src/screens/REPL.tsx`.
- Slash commands are registered in `src/commands.ts`.
- Tool implementations are registered through `src/tools.ts`.
- Model invocation is coordinated through `src/QueryEngine.ts` and `src/services/api/`.

## Validation

After changes, run:

```bash
bun run build:dev
./cli-dev auth status --json
```

For auth behavior, test an invalid key with:

```bash
./cli-dev auth login --api-key invalid-test-key
```

It should fail immediately and should not save the invalid key.

## License

Private internal repository. All rights reserved.
