// explained at https://rares.uk/typescript-unextend-interface/
export type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

export type NodeTypeType =
  | 'baseStats'
  | 'offensiveStats'
  | 'defensiveStats'
  | 'skillLvlStats'
  | 'passive'
  | 'specialStats';

export type PerkType = {
  name: string;
  type: NodeTypeType;
  fullNameList?: string[];
  value?: number;
  minValue?: number;
  maxValue?: number;
  description?: string;
};
type PerksType = PerkType[];

export type NODEType = {
  id: number;
  x: number;
  y: number;
  points: 0 | 1 | 2 | 3 | 4 | 5;
  type: 'basic' | 'notable' | 'keystone';
  /** Expended name for stats summary, use name if no fullNameList */
  perks: PerksType;
  /** Additional search keywords other than the name */
  additionalSearchKeywords?: string;
  linkedNodesIndexes: number[];
};

export type NODESType = NODEType[];

export interface NodeType extends NODEType {
  selectedPoints: number;
  isKeyNode?: boolean;
  isMSTNode?: boolean;
}
