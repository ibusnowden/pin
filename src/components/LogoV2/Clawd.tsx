import * as React from 'react';
import { Box, Text } from '../../ink.js';

export type ClawdPose =
  | 'default'
  | 'arms-up'
  | 'look-left'
  | 'look-right';

type Props = {
  pose?: ClawdPose;
};

const KIMI_ASCII = [
  'PPPP  III  N   N',
  'P   P  I   NN  N',
  'PPPP   I   N N N',
  'P      I   N  NN',
  'P     III  N   N',
] as const;

export function Clawd(_props: Props = {}): React.ReactNode {
  return (
    <Box flexDirection="column" alignItems="center">
      {KIMI_ASCII.map(line => (
        <Text key={line} color="clawd_body">
          {line}
        </Text>
      ))}
    </Box>
  );
}
