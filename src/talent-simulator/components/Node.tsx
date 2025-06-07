import React from 'react';
import { Text } from 'grommet';
import { withTranslation } from 'react-i18next';

import { SvgTooltip } from './SvgTooltip';

import { NodeType } from './types';

import { SHOW_PERCENTAGE } from './Constants';

export interface NodeProps extends NodeType {
  onAdd?: (selectedPoints?: number) => void;
  onRemove?: (selectedPoints?: number) => void;
  nodes: any;
  showTooltip?: boolean;
  searchString?: string;
  remindPoints?: number;
  t: any;
  isChinese: boolean;
  isKeyNode?: boolean;
  isMSTNode?: boolean;
  onNodeClick?: () => void;
}

const getTooltip = (
  { perks }: { type: string; points: number; perks: NodeType['perks'] },
  t: any,
  isChinese: boolean,
  showTooltip?: boolean
) => (
  <>
    {perks.map(({ name, value, minValue, maxValue, description }) => {
      const nameString = t(name);
      const valueString = value
        ? `+${SHOW_PERCENTAGE[name] ? `${Math.round(value * 10000) / 100}%` : value}`
        : minValue
        ? `+${minValue}~${maxValue}`
        : '';
      return (
        <React.Fragment key={name}>
          <Text>{isChinese ? `${nameString} ${valueString}` : `${valueString} ${nameString}`}</Text>
          {!showTooltip &&
            description &&
            t(description)
              .split('\n')
              .map((desc: string) => (
                <React.Fragment key={`${name}-${desc}`}>{desc ? <Text>{desc}</Text> : <br />}</React.Fragment>
              ))}
        </React.Fragment>
      );
    })}
  </>
);

const PureNode = (props: any) => (
  <>
    <circle {...props} />
    {props.type === 'notable' && (
      <polygon
        points={`0,0 ${props.r * 4},${props.r * 4}`}
        style={{ fill: 'lime', stroke: 'purple', strokeWidth: 1 }}
      />
    )}
    {props.type === 'keystone' && (
      <>
        <polygon
          points={`0,0 ${props.r * 4},${props.r * 4}`}
          style={{ fill: 'lime', stroke: 'purple', strokeWidth: 1 }}
        />
        <polygon
          points={`0,${props.r * 4} ${props.r * 4},0`}
          style={{ fill: 'lime', stroke: 'purple', strokeWidth: 1 }}
        />
      </>
    )}
  </>
);

export const Node = withTranslation('translations')(
  ({
    x,
    y,
    id,
    points,
    type,
    perks,
    additionalSearchKeywords,
    selectedPoints,
    nodes,
    linkedNodesIndexes,
    onAdd,
    onRemove,
    showTooltip,
    searchString,
    remindPoints,
    t,
    isChinese,
    isKeyNode,
    isMSTNode,
    onNodeClick,
  }: NodeProps) => {
    const circleRadius = 2.5;
    const sideLength = circleRadius * 4;

    let isPartiallySelected = false;
    let isFullySelected = false;
    let isOpen = false;
    let isFound = false;

    if (selectedPoints === points) {
      isFullySelected = true;
    } else if (selectedPoints) {
      isPartiallySelected = true;
    } else {
      isOpen =
        id <= 4 ||
        linkedNodesIndexes.find(nodeIndex => nodes[nodeIndex].selectedPoints === nodes[nodeIndex].points) !== undefined;
    }

    // TODO: Search should be done with special search key word string
    // so the compound node could be found correctly
    const _searchString = searchString?.toLocaleLowerCase();
    isFound =
      !!_searchString &&
      (additionalSearchKeywords
        ?.split(' ')
        .some(searchKeyword => searchKeyword && t(searchKeyword)?.toLocaleLowerCase().includes(_searchString)) ||
        perks.some(
          ({ name, type, fullNameList, description }) =>
            t(name).toLocaleLowerCase().includes(_searchString) ||
            t(type).toLocaleLowerCase().includes(_searchString) ||
            t(description)?.toLocaleLowerCase().includes(_searchString) ||
            fullNameList?.some(name => t(name).toLocaleLowerCase().includes(_searchString))
        ));

    const isAddable = (remindPoints ?? 0) > 0 && (isPartiallySelected || isOpen);

    const isRemovable =
      isPartiallySelected ||
      (isFullySelected &&
        ((id > 4 &&
          linkedNodesIndexes.filter(nodeIndex => {
            const linkedNode = nodes[nodeIndex];
            return linkedNode.selectedPoints;
          }).length < 2) ||
          linkedNodesIndexes.filter(nodeIndex => {
            const linkedNode = nodes[nodeIndex];
            return linkedNode.selectedPoints;
          }).length < 1));

    let nodeColor = 'LightGrey';
    if (isKeyNode) nodeColor = 'blue';
    else if (isMSTNode) nodeColor = 'green';

    return (
      <>
        {linkedNodesIndexes.map(nodeIndex => {
          if (nodeIndex > id) {
            const nextNode = nodes[nodeIndex];
            const isMSTEdge = isMSTNode && nextNode.isMSTNode;
            const isSelectedEdge = (isPartiallySelected && nextNode.selectedPoints === nextNode.points) ||
                                 (isFullySelected && nextNode.selectedPoints);
            const color =
              isMSTEdge
                ? 'green'
                : isSelectedEdge
                ? 'blue'
                : 'LightGrey';
            const strokeWidth = isMSTEdge ? 2 : 1;
            
            if (x === nextNode.x || y === nextNode.y) {
              return (
                <line
                  x1={x}
                  y1={y}
                  z={100}
                  x2={nextNode.x}
                  y2={nextNode.y}
                  stroke={color}
                  strokeWidth={strokeWidth}
                  key={`${id}-${nextNode.id}`}
                />
              );
            } else if (
              (linkedNodesIndexes.some(
                nodeIdx =>
                  // if there was a connected node two block or more above
                  (nodes[nodeIdx].x === x && nodes[nodeIdx].y <= y - 40) ||
                  // if there was a connected node on the left, right next to current
                  (nodes[nodeIdx].y === y && nodes[nodeIdx].x === x - 20 && nextNode.x < x)
              ) ||
                // if there was a non-connected node on the right, right next to current
                (nodes[id + 1].y === y && nodes[id + 1].x === x + 20 && nextNode.x > x) ||
                id === 35) &&
              id !== 218 &&
              id !== 186
            ) {
              // go top left/right with top first
              return (
                <React.Fragment key={`${id}-${nextNode.id}`}>
                  <line x1={x} y1={y} z={100} x2={x} y2={nextNode.y} stroke={color} strokeWidth={strokeWidth} />
                  <rect x={x - 0.5} y={nextNode.y - 0.5} width="1" height="1" fill={color} />
                  <line x1={x} y1={nextNode.y} z={100} x2={nextNode.x} y2={nextNode.y} stroke={color} strokeWidth={strokeWidth} />
                </React.Fragment>
              );
            } else {
              // go top left/right with left/right first
              return (
                <React.Fragment key={`${id}-${nextNode.id}`}>
                  <line x1={x} y1={y} z={100} x2={nextNode.x} y2={y} stroke={color} strokeWidth={strokeWidth} />
                  <rect x={nextNode.x - 0.5} y={y - 0.5} width="1" height="1" fill={color} />
                  <line x1={nextNode.x} y1={y} z={100} x2={nextNode.x} y2={nextNode.y} stroke={color} strokeWidth={strokeWidth} />
                </React.Fragment>
              );
            }
          }
          return null;
        })}
        <svg
          x={x - circleRadius * 2}
          y={y - circleRadius * 2}
          width={sideLength + 5}
          height={sideLength + 5}
          viewBox={`0 0 ${sideLength + 5} ${sideLength + 5}`}
          onClick={onNodeClick}
          style={{ cursor: 'pointer' }}
        >
          {isFound && <circle cx={circleRadius * 2} cy={circleRadius * 2} r={4} stroke={'red'} fillOpacity={0} />}
          {isRemovable && onRemove && (
            <rect
              x={9}
              y={0}
              width={4}
              height={2}
              fill="red"
              style={{ cursor: 'pointer' }}
              onClick={() => onRemove()}
            />
          )}
          <text
            x={7}
            y={12}
            fill={nodeColor}
            style={{
              fontSize: '5px',
              cursor: isAddable || isRemovable ? 'pointer' : 'default',
              // prevent text selection highlight
              WebkitTouchCallout: 'none', // iOS Safari
              WebkitUserSelect: 'none', // Safari
              MozUserSelect: 'none', // Old versions of Firefox
              msUserSelect: 'none', // Internet Explorer/Edge
              userSelect: 'none', // Non-prefixed version, currently supported by Chrome, Opera and Firefox
            }}
            onClick={
              isAddable && onAdd
                ? () => onAdd(Math.min(points - selectedPoints, remindPoints ?? 0))
                : isRemovable && onRemove
                ? () => onRemove(selectedPoints)
                : () => {}
            }>
            {selectedPoints}/{points}
          </text>
          <SvgTooltip tooltip={getTooltip(nodes[id], t, isChinese, showTooltip) || ''} forceShow={showTooltip}>
            <PureNode
              cx={circleRadius * 2}
              cy={circleRadius * 2}
              r={circleRadius}
              type={type}
              fill={nodeColor}
            />
          </SvgTooltip>
        </svg>
      </>
    );
  }
);
